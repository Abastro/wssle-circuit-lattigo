package wsslergsw

import (
	"errors"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Elect runs the Election phase (Fig. 1, Finalize line 12) on the aggregate
// from [Aggregate]. The aggregate encrypts sum_j Y^j f_j(Z) rotated by the
// joint randomness Y^r, where f_j(Z) is the commitment in slot j and
// Z = Y^W = X^(N/C): the winning slot lands on Y^0, that is on the subring
// Z[Z], and every other slot on some Y^j Z[Z], 0 < j < W. A single relative
// trace keeps the first and annihilates the rest,
//
//	ct_out = Tr_{R/Z[Z]}((N/C)^-1 * agg),
//
// so one decryption reveals the winner's C fragments, at X^(k*N/C)
// ([CircuitParams.FragmentIndex]), and nothing else. They come out multiplied
// by an unknown Z^c, which [DecodeFragments] undoes.
//
// The trace is over all of G_C, N/C automorphisms, rather than the W that
// would isolate the message within Z[Y]: the error occupies all of R, and only
// the relative trace of R itself is an exact map on it, Tr(e) = (N/C) times
// its Z[Z]-component for every e in R. That is what makes the (N/C)^-1
// legitimate: it cancels the trace's gain on both message and error, and the
// fragments come out at 2*Delta rather than 2*Delta*N/C. What the trace adds is
// its own key-switching error.
//
// There is no ciphertext-ciphertext multiplication: [Aggregate] has already
// folded in every party's randomness by external product, so there is no
// relinearization key anywhere in this variant.
func Elect(eval *rgsw.Evaluator, params CircuitParams, agg *rlwe.Ciphertext) *rlwe.Ciphertext {
	ringQ := params.RLWE.RingQ().AtLevel(agg.Level())

	gain := big.NewInt(int64(params.RLWE.N() / params.Fragments))
	inv := new(big.Int).ModInverse(gain, ringQ.Modulus())

	scaled := agg.CopyNew()
	ringQ.MulScalarBigint(scaled.Value[0], inv, scaled.Value[0])
	ringQ.MulScalarBigint(scaled.Value[1], inv, scaled.Value[1])

	out, err := Trace(&eval.Evaluator, scaled, ExtractionGaloisElements(params))
	if err != nil {
		panic(err)
	}
	return out
}

// DecodeFragments reads the elected commitment h* off the decrypted message of
// [Elect]'s output, its coefficients at X^(k*N/C) in fragment order, as
// [FinalDecrypt] returns them.
//
// They are the coefficients of Z^c h(Z) for an unknown c in [0, 2C), where
// h(Z) = sum_k (h_k + 1) Z^k holds the fragments offset by one: pass 2 rotates
// the winning slot past Y^W = Z
// some number of times, and each crossing multiplies it by Z, a negacyclic
// rotation of its fragments. The offset keeps every fragment strictly
// positive, so the rotation shows in the signs: for c < C the first c
// coefficients are negative and the rest positive, for c >= C the first c - C
// are positive and the rest negative. Those 2C patterns are distinct, which
// fixes c; h = Z^-c times the message, and h* = sum_k 2^(k*H) (h_k - 1).
//
// It fails when the pattern matches no rotation or a fragment falls outside
// [1, 2^H], which a correct decryption never produces.
func DecodeFragments(params CircuitParams, message []*big.Int) (*big.Int, error) {
	C := params.Fragments

	v := make([]*big.Int, C)
	negative := make([]bool, C)
	for k := range v {
		v[k] = new(big.Int).Set(message[k])
		if v[k].Sign() == 0 {
			return nil, errors.New("decoded fragment is zero: no rotation matches")
		}
		negative[k] = v[k].Sign() < 0
	}

	rotation := -1
	for c := 0; c < 2*C && rotation < 0; c++ {
		matches := true
		for k := 0; k < C; k++ {
			want := k < c // c < C: negative on the first c
			if c >= C {
				want = k >= c-C // c >= C: positive on the first c - C
			}
			if negative[k] != want {
				matches = false
				break
			}
		}
		if matches {
			rotation = c
		}
	}
	if rotation < 0 {
		return nil, errors.New("sign pattern matches no rotation of the fragments")
	}

	// Multiply by Z^-1, rotation times: coefficient k takes k+1, the last takes
	// minus the first (Z^C = -1).
	for ; rotation > 0; rotation-- {
		first := v[0]
		copy(v, v[1:])
		v[C-1] = first.Neg(first)
	}

	top := new(big.Int).Lsh(big.NewInt(1), params.FragmentBits) // 2^H
	frags := make([]*big.Int, C)
	for k, x := range v {
		if x.Sign() <= 0 || x.Cmp(top) > 0 {
			return nil, errors.New("decoded fragment outside [1, 2^H]")
		}
		frags[k] = new(big.Int).Sub(x, big.NewInt(1))
	}
	return JoinFragments(params, frags), nil
}

// DecodeResult is [DecodeFragments] on a plaintext decrypted under the whole
// key, which only the tests hold; an election decrypts through
// [PartialDecrypt] and [FinalDecrypt].
func DecodeResult(params CircuitParams, pt *rlwe.Plaintext) (*big.Int, error) {
	rounded := RoundCoeffs(params.RLWE, pt, params.ResultScale())
	message := make([]*big.Int, params.Fragments)
	for k := range message {
		message[k] = rounded[params.FragmentIndex(k)]
	}
	return DecodeFragments(params, message)
}
