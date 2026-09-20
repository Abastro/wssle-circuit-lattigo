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

// Ring/modulus sizing, matching the HIENAA reference parameters: a 104-bit
// modulus budget at N=2^12, reported there as meeting 128-bit security per
// the lattice-estimator (commit 14a3625), sized for 16-bit commitments and a
// total weight of up to 2^12.
//
// Unlike the CKKS variant of the same circuit, no multiplicative level chain
// is needed: the external product consumes no levels, so a single ciphertext
// prime suffices regardless of how many parties fold into [Aggregate].
const (
	logN          = 12
	hammingWeight = 256
	logQ          = 50
	logP          = 54
	logDelta      = 20
	sigma         = 3.2
)

// CircuitParams holds the ring parameters and packing layout for the circuit.
type CircuitParams struct {
	RLWE    rlwe.Parameters
	Delta   uint64 // fixed-point scaling factor applied to commitments.
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
		LogQ:    []int{logQ},
		LogP:    []int{logP},
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
