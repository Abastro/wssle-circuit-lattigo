package wsslergsw

import (
	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Identity is the accumulator [Aggregate] starts from: Enc(0). It carries no
// secret, but is built by a real encryption so that every party is treated
// symmetrically and none is special-cased as the seed of the fold.
func Identity(enc *rlwe.Encryptor, params CircuitParams) *rlwe.Ciphertext {
	ct, err := enc.EncryptNew(EncodeCoeffs(params.RLWE, make([]uint64, params.RLWE.N()), params.Delta))
	if err != nil {
		panic(err)
	}
	return ct
}

// Aggregate reduces all registrations to the single ciphertext [Elect] needs,
// Enc(2 * ecd_total * Y^(sum r_i)), in two sequential passes (Fig. 1, Finalize
// lines 6-11):
//
//	acc = acc (x) ct_W_i + encodeH(reg_i)   for every i, in party order
//	acc = acc (x) ct_R_i                    for every i
//
// The passes cannot be merged into one interleaved loop. Ring multiplication
// by a monomial is commutative, so applying every party's ct_R to the
// already-complete encoding shifts the whole thing by Y^(sum r_i) whatever
// the order; but applying a party's ct_R right after its own encoding step
// would undershift every term contributed by parties folded in earlier, which
// each need the *total* shift, including contributions that do not exist yet
// at that point in the pass.
//
// Note the first pass is order-sensitive (it determines which slot range each
// party's commitment occupies) though associative, and that seed is left
// untouched so it can be reused across elections. weights and regs are indexed
// in the same party order.
func Aggregate(eval *rgsw.Evaluator, seed *rlwe.Ciphertext, weights []*Weight, regs []*Registration) *rlwe.Ciphertext {
	params := eval.GetRLWEParameters()
	ringQ := params.RingQ().AtLevel(seed.Level())

	acc := seed.CopyNew()
	ecd := rlwe.NewCiphertext(params, 1, seed.Level())

	for i, reg := range regs {
		eval.ExternalProduct(acc, weights[i].CtW, acc)
		encodeH(eval, weights[i], reg, ecd)
		ringQ.Add(acc.Value[0], ecd.Value[0], acc.Value[0])
		ringQ.Add(acc.Value[1], ecd.Value[1], acc.Value[1])
	}
	for _, reg := range regs {
		eval.ExternalProduct(acc, reg.CtR, acc)
	}
	return acc
}

// encodeH builds a party's contribution ct_H = Enc(2*Delta*h*(Y^w - 1)/(Y - 1))
// from its registered constant Enc(Delta*h) and the public [Weight.CtEcd],
// writing it into dst.
//
// This is the step Fig. 1 has the party perform (line 5). Doing it here instead
// costs one external product per party and leaves the weight where it already
// was -- in the public parameters -- so ct_W and ct_H cannot disagree about w.
// The external product is taken in place, on a copy of ct_H. Its out-of-place
// form is broken for levelP >= 1: [rgsw.Evaluator.ExternalProduct] accumulates
// into its internal QP buffers but then rescales from (opOut.Value[i], buff.P),
// so the Q half of the result is dropped and replaced by whatever opOut already
// held (core/rgsw/evaluator.go:52 against :77). Copying first also carries the
// metadata, which the external product does not touch.
func encodeH(eval *rgsw.Evaluator, weight *Weight, reg *Registration, dst *rlwe.Ciphertext) {
	dst.Copy(reg.CtH)
	eval.ExternalProduct(dst, weight.CtEcd, dst)
}
