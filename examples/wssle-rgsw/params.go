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

// Ring/modulus sizing for 32-bit commitments, a total weight of up to 2^11,
// and the noise flooding that threshold decryption needs, semi-honest.
//
// The budget comes from the lattice estimator (the full LWE.estimate, not the
// rough one, which is hardcoded to usvp and dual_hybrid and so cannot even see
// bdd_mitm_hybrid, the attack that binds here): at N=2^13 with this sparse
// ternary secret, log(Q*P)=209 gives 128.5 bits, and each further bit costs
// 0.57.
//
// Within that budget everything is pinned by sigma_0, the noise at the constant
// coefficient of [Elect]'s output. It is derived, not measured: one election is
// a single sample of it, and the trace makes it sqrt(W) larger than the bulk it
// would otherwise be read off. What is measured is the three primitives, where
// one ciphertext carries N coefficients and so a handful of encryptions gives
// thousands of samples (see TestNoiseModel):
//
//	sigma_rlwe       = 2^2.21   fresh RLWE under pk
//	sigma_ext        = 2^2.23   one external product against an RGSW monomial
//	sigma_ext_tilde  = 2^3.59   one against the derived encoder, which carries
//	                            the ||2/(Y-1)||_2^2 = W amplification
//
// The chain of references/error-analysis then gives
//
//	sigma_ecd^2 <= (4W+1) sigma_rlwe^2 + n sigma_ext_tilde^2 + 2n sigma_ext^2
//	sigma_0     <= W sqrt(sigma_ecd^2 + sigma_ext^2/3)         = 2^20.55
//
// and with B = 7.24*sigma_0 bounding the evaluation error except w.p. 2^-41,
// and scale = 2*Delta*W = 2^64 -- the largest that fits, which must be checked
// exactly rather than as log2(Q) - 1 - log2(|h|_inf): Q falls 2^69.2 short of
// 2^98, so 2^65 overflows Q/2 by 2^67.8 for a commitment at the top of its
// range, while 2^64 clears it,
//
//	lambda = 36.7,
//
// with the smudging noise uniform on [-B_flood, B_flood] rather than Gaussian.
// The distinction is worth 2.6 bits here: a uniform smudge has a hard bound, so
// B_flood can be the whole rounding budget and decryption fails only when the
// circuit error exceeds B, i.e. with the same 2^-41 the security argument
// already assumes. A Gaussian must reserve 6.12 sigma_flood for its tail and
// still accepts a 2^-30 failure rate, which leaves only lambda = 38.0.
//
// sigma_rlwe is smaller than Definition 2 of the paper would give (2^6.18),
// because lattigo encrypts over QP and rescales by P, leaving the rounding term
// alone. That is a legitimate implementation choice, not an assumption of the
// analysis; sized against Definition 2 instead, the same modulus would support
// lambda = 34.8.
//
// Q=98 is the best split of the 209. Above it sigma_0 climbs as (Q/P)^2 through
// sigma_ext_tilde; below it sigma_0 has already reached its floor of 2^20.12,
// set by h alone, so a bit off Q is a bit of scale lost for nothing. lambda=40
// is out of reach at this ring: it needs log(Q*P) = 215, costing 3.4 bits of
// lattice security. N=2^14 clears it with room to spare.
//
// Unlike the CKKS variant of the same circuit, no multiplicative level chain is
// needed: the external product consumes no levels, so the modulus is sized
// purely by the noise budget, not by how many parties fold into [Aggregate].
const (
	logN          = 13
	hammingWeight = 256
	logDelta      = 52
	sigma         = 3.2
)

// Q and P, split into NTT-friendly primes. dnum = ceil(len(logQ)/len(logP)) = 1.
var (
	logQ = []int{49, 49} // 98 bits
	logP = []int{56, 55} // 111 bits
)

// CircuitParams holds the ring parameters and packing layout for the circuit.
type CircuitParams struct {
	RLWE rlwe.Parameters
	// Delta is the scaling factor applied to commitments. It is not a free
	// knob: [CircuitParams.ResultScale] has to leave room for the flooding
	// noise, which is what sizes it here (see the sizing note above).
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
// threshold decryption needs puts logDelta at 52, so the scale is 2^64.
func (p CircuitParams) ResultScale() *big.Int {
	return new(big.Int).Mul(p.EncodedScale(), new(big.Int).SetUint64(p.TotalWt))
}

// SetupParams builds [CircuitParams] for an election with the given public
// total weight, which must be a power of two dividing the ring degree, on the
// shipped ring and modulus (parameter set C of the paper).
func SetupParams(totalWeight uint64) CircuitParams {
	return NewCircuitParams(logN, logQ, logP, logDelta, totalWeight)
}

// NewCircuitParams builds [CircuitParams] on an explicit ring and modulus, for
// the parameter sets other than the shipped one. The sizing argument above
// applies to each; benchmarks/09-three-parameter-sets.txt records the three.
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
