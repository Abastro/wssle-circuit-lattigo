// Package wsslergsw implements the RGSW variant of the WSSLE ("Pastel")
// single secret leader election circuit -- Fig. 1, p.18 of
// references/Weighted_Single_Secret_Leader_Election_using_CKKS.pdf -- on top
// of Lattigo's core RLWE/RGSW primitives.
//
// It is a port of the HIENAA reference implementation (examples/wssle-rgsw
// there), kept deliberately close to it so the two can be benchmarked against
// each other: same ring degree, same modulus sizes, same secret distribution,
// same circuit shape, same test scenarios.
//
// Everything is coefficient-encoded at the bare [rlwe] level rather than
// through a scheme package: the circuit is only Add, external product and
// automorphism, so there is no rescaling, no relinearization and no
// scheme-level scaling-factor bookkeeping to carry. Messages are plain
// integers scaled by a fixed [CircuitParams.Delta].
//
// Threshold decryption/DKG (ThFHE.Setup/ThFHE.Dec) and the NIZKs of the
// malicious-security protocol are out of scope here, exactly as in the
// reference implementation: a single secret key stands in for the whole key
// committee.
package wsslergsw

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// Each parameter set is sized by two constraints on the scale S = 2*Delta*W
// that [Elect]'s output carries, one from each side (references/error-analysis,
// sec:dec):
//
//	from below   (2|K| 2^s + 1) * beta * sigma_0  <=  S/2
//	from above   (2^H - 1) * S  +  S/2            <=  Q/2
//
// The lower bound is the flooding budget of threshold decryption. Each of the
// |K| = 32 committee members adds a sum of two uniforms on [-2^s B, 2^s B] with
// s = 32: that masks the evaluation error to statistical distance 2^-64 (Dahl
// et al., WAHC'23, eprint 2023/815), and, having bounded support, bounds the
// flooding outright. So the only event that can fail decryption is
// |e_eval| > B = beta*sigma_0, which beta = 10.08 (10.15 at N = 2^14) makes a
// 2^-64 event after a union bound over the N coefficients.
//
// sigma_0 is the noise at the constant coefficient, the one the trace
// amplifies coherently. It is derived, not measured -- an election is a single
// sample of it -- from the measured primitive variances sigma_rlwe, sigma_ext
// and sigma_ext_tilde, each over a fresh key pair and a fresh RGSW operand per
// sample (see TestFloodingBudget).
//
// The upper bound keeps the elected commitment from wrapping Q. It caps S at
// 2^floor(log2 Q - H - 1), with log2 Q the actual size of the prime product:
// lattigo's primes land a little above or below their nominal sizes, which
// moves the floor by one when log2 Q - H - 1 falls just short of an integer, as
// it does for set C. [CircuitParams.MaxCommitment] reports the bound exactly.
//
// Lattice security is from the full LWE.estimate of the lattice estimator, not
// the rough one, which runs only usvp and dual_hybrid and so misses
// bdd_mitm_hybrid, the attack that binds for this sparse ternary secret:
//
//	set  log(QP)  security  sigma_0   flooding headroom
//	A    416      128.0     2^24.8    +3.8 bits
//	B    210      127.9     2^22.7    +1.0 bits
//	C    209      128.5     2^20.8    +0.9 bits
//
// Set B is the one whose split matters. With d = 2 the decomposition digit has
// to stay well under P, while the ceiling wants Q large for its 64-bit
// coefficients; at log(QP) = 209 the best split measured only +0.1 bits. One
// more bit, on P, buys the +1.0.
//
// Unlike the CKKS variant of the same circuit, no multiplicative level chain is
// needed: the external product consumes no levels, so the modulus is sized
// purely by the noise budget, not by how many parties fold into [Aggregate].
const (
	hammingWeight = 256
	sigma         = 3.2
)

// ParamSet is one row of the paper's parameter table.
type ParamSet struct {
	Name        string
	LogN        int
	LogQ, LogP  []int // bit sizes of the Q and P primes; d = ceil(len(LogQ)/len(LogP))
	LogDelta    int
	TotalWeight uint64 // W
	// CoeffBits is H, the bit size of the value one coefficient carries. The
	// paper splits a commitment over C coefficients; this implementation
	// places each commitment in a single coefficient, so its commitments are
	// CoeffBits wide.
	CoeffBits uint
}

// The three parameter sets of the paper.
var (
	ParamSetA = ParamSet{
		Name: "A", LogN: 14,
		LogQ: []int{50, 50, 50, 50}, LogP: []int{54, 54, 54, 54}, // 200 + 216, d = 1
		LogDelta: 56, TotalWeight: 1 << 14, CoeffBits: 128,
	}
	ParamSetB = ParamSet{
		Name: "B", LogN: 13,
		LogQ: []int{33, 33, 33, 32}, LogP: []int{40, 39}, // 131 + 79, d = 2
		LogDelta: 53, TotalWeight: 1 << 12, CoeffBits: 64,
	}
	ParamSetC = ParamSet{
		Name: "C", LogN: 13,
		LogQ: []int{49, 49}, LogP: []int{56, 55}, // 98 + 111, d = 1
		LogDelta: 52, TotalWeight: 1 << 11, CoeffBits: 32,
	}

	ParamSets = []ParamSet{ParamSetA, ParamSetB, ParamSetC}
)

// Params builds the [CircuitParams] of the set.
func (ps ParamSet) Params() CircuitParams {
	return NewCircuitParams(ps.LogN, ps.LogQ, ps.LogP, ps.LogDelta, ps.TotalWeight)
}

// CircuitParams holds the ring parameters and packing layout for the circuit.
type CircuitParams struct {
	RLWE rlwe.Parameters
	// Delta is the scaling factor applied to commitments. It is not a free
	// knob: [CircuitParams.ResultScale] has to leave room for the flooding
	// noise below and [CircuitParams.MaxCommitment] above (see [ParamSet]).
	Delta   uint64
	Stride  int    // S = N/W, the packing stride: Y = X^S generates the subring.
	TotalWt uint64 // W, the public total weight.
}

// Scale is the factor a freshly encrypted commitment carries: Delta alone.
func (p CircuitParams) Scale() *big.Int {
	return new(big.Int).SetUint64(p.Delta)
}

// EncodedScale is the factor a party's contribution carries once [encodeH] has
// applied the 2/(Y-1) of [deriveEncoder]: 2*Delta.
func (p CircuitParams) EncodedScale() *big.Int {
	return new(big.Int).Lsh(p.Scale(), 1)
}

// ResultScale is the factor [Elect]'s output carries on the elected
// commitment: Delta from the encoding, 2 from 2/(Y-1), and W from the trace.
// All three are divided out in the clear, which is exact and free -- the
// election result is public.
//
// It is a [big.Int] because it does not fit a uint64: the flooding noise that
// threshold decryption needs takes it to 2^64 and beyond (2^71 for set A).
func (p CircuitParams) ResultScale() *big.Int {
	return new(big.Int).Mul(p.EncodedScale(), new(big.Int).SetUint64(p.TotalWt))
}

// MaxCommitment is the largest commitment [Elect]'s output can carry without
// wrapping Q: the largest h with h*S + S/2 <= Q/2, S = [CircuitParams.ResultScale],
// where S/2 is the most error the final rounding tolerates. [Register] rejects
// anything larger.
func (p CircuitParams) MaxCommitment() *big.Int {
	scale := p.ResultScale()
	h := new(big.Int).Rsh(p.RLWE.RingQ().Modulus(), 1) // floor(Q/2): Q is odd
	h.Sub(h, new(big.Int).Rsh(scale, 1))
	return h.Div(h, scale)
}

// SetupParams builds [CircuitParams] on set C's ring and modulus for an
// election of any other public total weight, which must be a power of two
// dividing the ring degree. The step tests use it with small W.
func SetupParams(totalWeight uint64) CircuitParams {
	c := ParamSetC
	return NewCircuitParams(c.LogN, c.LogQ, c.LogP, c.LogDelta, totalWeight)
}

// NewCircuitParams builds [CircuitParams] on an explicit ring and modulus.
func NewCircuitParams(logN int, logQ, logP []int, logDelta int, totalWeight uint64) CircuitParams {
	N := 1 << logN
	if totalWeight == 0 || N%int(totalWeight) != 0 {
		panic("total weight must divide the ring degree")
	}

	params, err := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN:    logN,
		LogQ:    logQ,
		LogP:    logP,
		Xs:      ring.Ternary{H: hammingWeight},
		Xe:      ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma},
		NTTFlag: true,
	})
	if err != nil {
		panic(err)
	}

	return CircuitParams{
		RLWE:    params,
		Delta:   1 << logDelta,
		Stride:  N / int(totalWeight),
		TotalWt: totalWeight,
	}
}

// SetupKeys generates the secret key, the public key parties encrypt under,
// and the automorphism keys [Trace] needs.
//
// There is no relinearization key: this variant never multiplies two
// ciphertexts anywhere (see [Elect]), so Galois keys are the only evaluation
// keys in play.
//
// The keys are generated centrally from a single secret key, standing in for
// the key committee's DKG. Under a real N-out-of-N setup the same keys would
// come from multiparty.GaloisKeyGenProtocol (or from a trusted dealer holding
// the summed secret key); neither changes the shape or the cost of what the
// circuit below evaluates.
//
// Registration must encrypt under pk, not sk: a party does not hold the secret
// key. The distinction is not cosmetic for the noise. A secret-key RGSW
// ciphertext carries error variance sigma_err^2, whereas a public-key one
// carries sigma_err^2*(1 + N*sigma_Enc^2 + N*sigma_sk^2) = sigma_err^2*(2h+1),
// a factor of 513 at h=256 -- 4.5 bits of sigma, straight through the circuit
// and onto the modulus (see the sizing note above).
func SetupKeys(params CircuitParams) (*rlwe.SecretKey, *rlwe.PublicKey, rlwe.EvaluationKeySet) {
	kgen := rlwe.NewKeyGenerator(params.RLWE)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	gks := kgen.GenGaloisKeysNew(TraceGaloisElements(2*int(params.TotalWt)), sk)
	return sk, pk, rlwe.NewMemEvaluationKeySet(nil, gks...)
}
