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
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// Ring/modulus sizing for 32-bit commitments and a total weight of up to
// 2^11, with headroom for the noise flooding that threshold decryption needs.
//
// N=2^13 admits roughly a 208-bit Q*P budget at 128-bit security. The
// correctness requirement is
//
//	log Q >= 1 + log h + log(2*beta) + lambda + log sigma
//
// (see references/error-analysis): 1 bit for the centred lift, 32 for the
// commitment, log(2*beta)+log sigma ~= 30 for a 6-sigma bound on the decryption
// noise, and lambda=40 for smudging. That is 103 bits of Q, leaving 105 for P.
//
// P >= Q keeps the gadget at a single digit (dnum=1), which is both the fastest
// and the quietest geometry: shrinking P to buy Q budget costs a one-time ~5
// bits of noise and multiplies the per-external-product cost by dnum.
//
// Unlike the CKKS variant of the same circuit, no multiplicative level chain is
// needed: the external product consumes no levels, so the modulus is sized
// purely by the noise budget, not by how many parties fold into [Aggregate].
const (
	logN          = 13
	hammingWeight = 256
	logDelta      = 24
	sigma         = 3.2
)

// Q and P, split into NTT-friendly primes. dnum = ceil(len(logQ)/len(logP)) = 1.
var (
	logQ = []int{52, 51} // 103 bits
	logP = []int{53, 52} // 105 bits
)

// CircuitParams holds the ring parameters and packing layout for the circuit.
type CircuitParams struct {
	RLWE rlwe.Parameters
	// Delta is the scaling factor applied to commitments. It is not a free
	// knob: correct rounding needs Delta*W > 2*beta*sigma, where sigma is the
	// noise at the constant coefficient after [Elect]. logDelta=24 puts
	// Delta*W at 2^35 against a measured requirement of 2^30.
	Delta   uint64
	Stride  int    // S = N/W, the packing stride: Y = X^S generates the subring.
	TotalWt uint64 // W, the public total weight.
}

// SetupParams builds [CircuitParams] for an election with the given public
// total weight, which must be a power of two dividing the ring degree.
func SetupParams(totalWeight uint64) CircuitParams {
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

// SetupKeys generates the secret key and the automorphism keys [Trace] needs.
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
func SetupKeys(params CircuitParams) (*rlwe.SecretKey, rlwe.EvaluationKeySet) {
	kgen := rlwe.NewKeyGenerator(params.RLWE)
	sk := kgen.GenSecretKeyNew()
	gks := kgen.GenGaloisKeysNew(TraceGaloisElements(2*int(params.TotalWt)), sk)
	return sk, rlwe.NewMemEvaluationKeySet(nil, gks...)
}
