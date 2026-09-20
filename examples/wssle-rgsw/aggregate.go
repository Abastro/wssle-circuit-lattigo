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
// Enc(ecd_total * Y^(sum r_i)), in two sequential passes (Fig. 1, Finalize
// lines 6-11):
//
//	acc = acc (x) ct_W_i + ct_H_i   for every i, in party order
//	acc = acc (x) ct_R_i            for every i
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
// untouched so it can be reused across elections.
func Aggregate(eval *rgsw.Evaluator, seed *rlwe.Ciphertext, regs []*Registration) *rlwe.Ciphertext {
	ringQ := eval.GetRLWEParameters().RingQ().AtLevel(seed.Level())

	acc := seed.CopyNew()
	for _, reg := range regs {
		eval.ExternalProduct(acc, reg.CtW, acc)
		ringQ.Add(acc.Value[0], reg.CtH.Value[0], acc.Value[0])
		ringQ.Add(acc.Value[1], reg.CtH.Value[1], acc.Value[1])
	}
	for _, reg := range regs {
		eval.ExternalProduct(acc, reg.CtR, acc)
	}
	return acc
}
