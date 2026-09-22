package wsslergsw

import (
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Identity is a fresh encryption of zero under the encryptor's key: the input
// the noise measurements run their external products on.
func Identity(enc *rlwe.Encryptor, params CircuitParams) *rlwe.Ciphertext {
	ct, err := enc.EncryptNew(EncodeCoeffs(params.RLWE, make([]uint64, params.RLWE.N()), params.Delta))
	if err != nil {
		panic(err)
	}
	return ct
}

// Helpers for the noise measurements of TestFloodingBudget.

func sq(x float64) float64 { return x * x }

// lg reports a variance in bits of standard deviation.
func lg(variance float64) float64 { return 0.5 * math.Log2(variance) }

// rms is the root-mean-square of a residual vector.
func rms(res []*big.Int) float64 {
	sum := new(big.Float)
	for _, v := range res {
		f := new(big.Float).SetInt(v)
		sum.Add(sum, f.Mul(f, f))
	}
	ms, _ := sum.Quo(sum, big.NewFloat(float64(len(res)))).Float64()
	return math.Sqrt(ms)
}
