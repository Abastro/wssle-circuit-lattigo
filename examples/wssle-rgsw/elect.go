package wsslergsw

import (
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Elect runs the Election phase (Fig. 1, Finalize line 12): ct_out <- tr(agg).
//
// There is no ciphertext-ciphertext multiplication left to do: [Aggregate]
// has already folded in every party's randomness by external product, so agg
// encrypts ecd_total * Y^(sum r_i) -- exactly the value the paper obtains as
// the ct_ecd (x) ct_r tensor. Hence no relinearization key anywhere in this
// variant.
//
// The trace's factor of W is not removed here; [DecodeResult] divides it out
// in the clear after decryption (see [DecodeCoeffs]).
func Elect(eval *rgsw.Evaluator, agg *rlwe.Ciphertext, totalWeight uint64) *rlwe.Ciphertext {
	ct, err := Trace(&eval.Evaluator, agg, 2*int(totalWeight))
	if err != nil {
		panic(err)
	}
	return ct
}

// DecodeResult reads the elected commitment h* off [Elect]'s decrypted
// output: it undoes the fixed-point scaling and the trace's factor W, then
// returns the magnitude of the single nonzero coefficient.
//
// The magnitude, per Fig. 1 line 16 (h* <- W^-1 |h'|): the winning
// coefficient can come back negated by the ring's negacyclic wraparound, when
// the accumulated random shift carries it past Y^W = -1.
func DecodeResult(params CircuitParams, pt *rlwe.Plaintext) uint64 {
	for _, v := range DecodeCoeffs(params.RLWE, pt, params.Delta*params.TotalWt) {
		if r := math.Round(v); r != 0 {
			return uint64(math.Abs(r))
		}
	}
	return 0
}
