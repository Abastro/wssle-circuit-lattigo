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
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// Each parameter set is sized by two constraints on the scale S = 2*Delta that
// every fragment of [Elect]'s output carries, one from each side
// (references/error-analysis, sec:dec):
//
//	from below   (2|K| 2^s + 1) * beta * sigma_0  <=  S/2
//	from above   (2^H - 1) * S  +  S/2            <=  Q/2
//
// The lower bound is the flooding budget of threshold decryption (see
// [PartialDecrypt]). Each of the |K| = 32 committee members adds, to each of
// the C output coefficients it decrypts, a sum of two uniforms on
// [-2^s B, 2^s B]: that masks the evaluation error to statistical distance
// 2^(-2s) per coefficient (Dahl et al., WAHC'23, eprint 2023/815), so
// s = ceil((64 + log2 C)/2) hides a whole share to 2^-64, and, having bounded
// support, it bounds the flooding outright. So the only event that can fail
// decryption is |e_eval| > B = beta*sigma_0, which beta = 9.16 to 9.30 makes a
// 2^-64 event after a union bound over the C decrypted coefficients.
//
// sigma_0 is the noise at an output fragment. [Elect] extracts each fragment
// with the full trace of the ring, pre-multiplied by N^-1 mod Q: the trace is an
// exact Z-linear map with Tr(e) = N*e_0 for every e in R, noise included, so
// the pre-multiplication cancels its gain on the aggregate's error outright,
// and what the trace adds is its own key switching, which the later doubling
// steps amplify:
//
//	sigma_0^2 = sigma_ecd^2 + (N^2 - 1)/3 * sigma_ks^2
//
// It is derived, not measured -- an election is a single sample of it -- from
// the measured primitive variances sigma_rlwe, sigma_ext and sigma_ext_tilde,
// each over a fresh key pair and a fresh RGSW operand per sample, with
// sigma_ks^2 taken as sigma_ext^2 (see TestFloodingBudget).
//
// The upper bound keeps each fragment from wrapping Q. It caps S at
// 2^floor(log2 Q - H - 1), with log2 Q the actual size of the prime product:
// lattigo's primes land a little above or below their nominal sizes, which
// moves the floor by one when log2 Q - H - 1 falls just short of an integer, as
// it does for set C. [CircuitParams.MaxFragment] reports the bound exactly.
//
// Lattice security is from the full LWE.estimate of the lattice estimator, not
// the rough one, which runs only usvp and dual_hybrid and so misses
// bdd_mitm_hybrid, the attack that binds for this sparse ternary secret:
//
//	set  log(QP)  security  C x H   sigma_0   flooding headroom
//	A    416      128.0     1x128   2^15.4    +13.2 bits
//	B    210      127.9     2x64    2^14.4    +9.2 bits
//	C    209      128.5     4x32    2^14.4    +7.2 bits
//
// In every set it is the trace's key switching that sets sigma_0; the
// aggregate's own error, 2^9.8 to 2^10.9, is well below it.
//
// Unlike the CKKS variant of the same circuit, no multiplicative level chain is
// needed: the external product consumes no levels, so the modulus is sized
// purely by the noise budget, not by how many parties fold into [Aggregate].
const (
	hammingWeight = 256
	sigma         = 3.2

	// statSecurity is the statistical distance, 2^-statSecurity, to which a
	// decryption share hides the evaluation error; the same 64 bits bound the
	// decryption failure probability through B (see [ParamSet.LogErrorBound]).
	statSecurity = 64
)

// ParamSet is one row of the paper's parameter table.
type ParamSet struct {
	Name        string
	LogN        int
	LogQ, LogP  []int // bit sizes of the Q and P primes; d = ceil(len(LogQ)/len(LogP))
	LogDelta    int
	TotalWeight uint64 // W
	// A commitment is split into Fragments (the paper's C) fragments of
	// FragmentBits (its H) bits each, the k-th carried by the coefficient of
	// X^(S*j + k) of its slot j. Fragments must not exceed the stride S.
	Fragments    int
	FragmentBits uint

	// CommitteeSize is m, the size of the key committee, which is separate from
	// the parties and decrypts m-out-of-m.
	CommitteeSize int
	// LogErrorBound is log2 B, a bound on the evaluation error at an output
	// fragment holding except with probability 2^-64: B >= beta*sigma_0 with
	// C*erfc(beta/sqrt2) = 2^-64, sigma_0 as derived by TestFloodingBudget,
	// rounded up to a power of two. It sizes the flooding.
	LogErrorBound int
}

// The three parameter sets of the paper, all for 128-bit commitments and a
// committee of 32.
var (
	ParamSetA = ParamSet{
		Name: "A", LogN: 14,
		LogQ: []int{50, 50, 50, 50}, LogP: []int{54, 54, 54, 54}, // 200 + 216, d = 1
		LogDelta: 70, TotalWeight: 1 << 14, Fragments: 1, FragmentBits: 128,
		CommitteeSize: 32, LogErrorBound: 19, // B >= 9.155 * 2^15.42 = 2^18.61
	}
	ParamSetB = ParamSet{
		Name: "B", LogN: 13,
		LogQ: []int{33, 33, 33, 32}, LogP: []int{40, 39}, // 131 + 79, d = 2
		LogDelta: 65, TotalWeight: 1 << 12, Fragments: 2, FragmentBits: 64,
		CommitteeSize: 32, LogErrorBound: 18, // B >= 9.230 * 2^14.43 = 2^17.64
	}
	ParamSetC = ParamSet{
		Name: "C", LogN: 13,
		LogQ: []int{49, 49}, LogP: []int{56, 55}, // 98 + 111, d = 1
		LogDelta: 63, TotalWeight: 1 << 11, Fragments: 4, FragmentBits: 32,
		CommitteeSize: 32, LogErrorBound: 18, // B >= 9.304 * 2^14.43 = 2^17.65
	}

	ParamSets = []ParamSet{ParamSetA, ParamSetB, ParamSetC}
)

// Params builds the [CircuitParams] of the set.
func (ps ParamSet) Params() CircuitParams {
	N := 1 << ps.LogN
	W := ps.TotalWeight
	if W == 0 || N%int(W) != 0 {
		panic("total weight must divide the ring degree")
	}
	if ps.Fragments < 1 || ps.Fragments > N/int(W) {
		panic("fragments per commitment must lie in [1, N/W]")
	}

	params, err := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN:    ps.LogN,
		LogQ:    ps.LogQ,
		LogP:    ps.LogP,
		Xs:      ring.Ternary{H: hammingWeight},
		Xe:      ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma},
		NTTFlag: true,
	})
	if err != nil {
		panic(err)
	}

	cp := CircuitParams{
		RLWE:          params,
		Delta:         new(big.Int).Lsh(big.NewInt(1), uint(ps.LogDelta)),
		Stride:        N / int(W),
		TotalWt:       W,
		Fragments:     ps.Fragments,
		FragmentBits:  ps.FragmentBits,
		CommitteeSize: ps.CommitteeSize,
		LogErrorBound: ps.LogErrorBound,
	}
	if ps.CommitteeSize < 1 {
		panic("the key committee needs at least one member")
	}
	if !cp.FloodingFits() {
		panic("the flooding of a full committee does not fit the output scale")
	}
	return cp
}

// SetupParams builds [CircuitParams] on set C's ring, modulus and fragment
// layout for an election of any other public total weight, which must be a
// power of two with N/W at least set C's four fragments. The step tests use it
// with small W.
func SetupParams(totalWeight uint64) CircuitParams {
	c := ParamSetC
	c.TotalWeight = totalWeight
	return c.Params()
}

// CircuitParams holds the ring parameters and packing layout for the circuit.
type CircuitParams struct {
	RLWE rlwe.Parameters
	// Delta is the scaling factor applied to commitment fragments. It is not a
	// free knob: [CircuitParams.ResultScale] has to leave room for the flooding
	// noise below and [CircuitParams.MaxFragment] above (see [ParamSet]).
	Delta   *big.Int
	Stride  int    // S = N/W, the packing stride: Y = X^S generates the subring.
	TotalWt uint64 // W, the public total weight.

	Fragments    int  // C, the fragments per commitment
	FragmentBits uint // H, the bits per fragment

	CommitteeSize int // m, the key committee
	LogErrorBound int // log2 B, see [ParamSet.LogErrorBound]
}

// CommitmentBits is the width of a commitment, C*H.
func (p CircuitParams) CommitmentBits() uint {
	return uint(p.Fragments) * p.FragmentBits
}

// Scale is the factor a freshly encrypted fragment carries: Delta alone.
func (p CircuitParams) Scale() *big.Int {
	return new(big.Int).Set(p.Delta)
}

// EncodedScale is the factor a party's contribution carries once [encodeH] has
// applied the 2/(Y-1) of [deriveEncoder]: 2*Delta.
func (p CircuitParams) EncodedScale() *big.Int {
	return new(big.Int).Lsh(p.Delta, 1)
}

// ResultScale is the factor each fragment of [Elect]'s output carries: the
// same 2*Delta as the aggregate. The trace would multiply it by N, but [Elect]
// pre-multiplies by N^-1 mod Q, which cancels that exactly.
func (p CircuitParams) ResultScale() *big.Int {
	return p.EncodedScale()
}

// SmudgeBits is s, the ratio 2^s of the flooding range to the error bound B.
// A share reveals C coefficients, each hidden to statistical distance 2^(-2s)
// by the two uniforms of [PartialDecrypt]; the share as a whole is then within
// C * 2^(-2s) <= 2^-statSecurity.
func (p CircuitParams) SmudgeBits() int {
	return int(math.Ceil((statSecurity + math.Log2(float64(p.Fragments))) / 2))
}

// FloodBound is F = 2^s * B: each member floods each coefficient of its share
// with the sum of two uniforms on [-F, F].
func (p CircuitParams) FloodBound() *big.Int {
	return new(big.Int).Lsh(big.NewInt(1), uint(p.LogErrorBound+p.SmudgeBits()))
}

// FloodingFits reports whether the whole committee's flooding, together with
// the evaluation error, stays below the S/2 that rounding tolerates:
//
//	2m*F + B <= S/2.
//
// The left side is a hard bound -- uniforms have bounded support -- except for
// B itself, which the evaluation error exceeds with probability 2^-64. So a
// set passing this check fails decryption only in that event.
func (p CircuitParams) FloodingFits() bool {
	lhs := new(big.Int).Mul(p.FloodBound(), big.NewInt(int64(2*p.CommitteeSize)))
	lhs.Add(lhs, new(big.Int).Lsh(big.NewInt(1), uint(p.LogErrorBound)))
	return lhs.Cmp(new(big.Int).Rsh(p.ResultScale(), 1)) <= 0
}

// MaxFragment is the largest fragment [Elect]'s output can carry without
// wrapping Q: the largest h with h*S + S/2 <= Q/2, S = [CircuitParams.ResultScale],
// where S/2 is the most error the final rounding tolerates. The same bound
// covers the aggregate, whose coefficients carry fragments at the same scale.
func (p CircuitParams) MaxFragment() *big.Int {
	scale := p.ResultScale()
	h := new(big.Int).Rsh(p.RLWE.RingQ().Modulus(), 1) // floor(Q/2): Q is odd
	h.Sub(h, new(big.Int).Rsh(scale, 1))
	return h.Div(h, scale)
}

// SetupKeys generates the secret key, the public key parties encrypt under,
// and the automorphism keys of the full trace [Elect] applies.
//
// There is no relinearization key: this variant never multiplies two
// ciphertexts anywhere (see [Elect]), so Galois keys are the only evaluation
// keys in play.
//
// This is the trusted dealer: it generates every key from one secret key, and
// then hands the committee additive shares of that key ([ShareSecretKey]) and
// erases it. The circuit only ever sees pk and the evaluation keys; decryption
// goes through the shares.
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
	gks := kgen.GenGaloisKeysNew(TraceGaloisElements(2*params.RLWE.N()), sk)
	return sk, pk, rlwe.NewMemEvaluationKeySet(nil, gks...)
}
