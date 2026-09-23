package wsslergsw

import (
	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
)

// Aggregate reduces all registrations to the single ciphertext [Elect] needs,
// Enc(2 * sum_j Y^j f_j(Z) * Y^(sum r_i)), in two sequential passes (Fig. 1,
// Finalize lines 6-11), starting from the first party's own contribution:
//
//	acc = encodeH(reg_1)
//	acc = acc (x) ct_W_i + encodeH(reg_i)   for i = 2 .. n, in party order
//	acc = acc (x) ct_R_i                    for i = 1 .. n
//
// The passes cannot be merged into one interleaved loop. Ring multiplication
// by a monomial is commutative, so applying every party's ct_R to the
// already-complete encoding shifts the whole thing by Y^(sum r_i) whatever
// the order; but applying a party's ct_R right after its own encoding step
// would undershift every term contributed by parties folded in earlier, which
// each need the *total* shift, including contributions that do not exist yet
// at that point in the pass.
//
// The first pass is order-sensitive (it determines which slot range each
// party's commitment occupies) though associative. weights and regs are
// indexed in the same party order.
func Aggregate(eval *rgsw.Evaluator, params CircuitParams, weights []*rgsw.Ciphertext, regs []*Registration) *rlwe.Ciphertext {
	p := params.RLWE
	level := regs[0].CtH.Level()
	ringQ := p.RingQ().AtLevel(level)

	// theta and the encoder buffer are shared by every party: theta does not
	// depend on the weight, and the derived encoder is consumed by the
	// external product of encodeH before the next one overwrites it.
	levelQ, levelP := weights[0].LevelQ(), weights[0].LevelP()
	theta := thetaQP(params, p.RingQP().AtLevel(levelQ, levelP))
	ctEcd := rgsw.NewCiphertext(p, levelQ, levelP, 0)

	acc := rlwe.NewCiphertext(p, 1, level)
	deriveEncoder(params, theta, weights[0], ctEcd)
	encodeH(eval, ctEcd, regs[0], acc)

	ecd := rlwe.NewCiphertext(p, 1, level)
	for i := 1; i < len(regs); i++ {
		eval.ExternalProduct(acc, weights[i], acc)
		deriveEncoder(params, theta, weights[i], ctEcd)
		encodeH(eval, ctEcd, regs[i], ecd)
		ringQ.Add(acc.Value[0], ecd.Value[0], acc.Value[0])
		ringQ.Add(acc.Value[1], ecd.Value[1], acc.Value[1])
	}
	for _, reg := range regs {
		eval.ExternalProduct(acc, reg.CtR, acc)
	}
	return acc
}

// encodeH builds a party's contribution ct_H = Enc(2*Delta*h(Z)*(Y^w - 1)/(Y - 1))
// from its registered Enc(Delta*h(Z)) and the encoder ctEcd that
// [deriveEncoder] produced from the party's public weight, writing it into dst.
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
func encodeH(eval *rgsw.Evaluator, ctEcd *rgsw.Ciphertext, reg *Registration, dst *rlwe.Ciphertext) {
	dst.Copy(reg.CtH)
	eval.ExternalProduct(dst, ctEcd, dst)
}

// deriveEncoder turns RGSW(Y^w) into RGSW(2*(Y^w - 1)/(Y - 1)), written into
// dst, by public arithmetic alone, in two steps:
//
//	ct - p*G                subtract the trivial encryption of 1
//	theta * (ct - p*G)      multiply by the plaintext theta = 2/(Y - 1)
//
// Both are exact on the message. The first leaves the error untouched: p*G is
// the whole 2d x 2 gadget matrix of Definition 3, whose rows have phases p*g_i
// and p*g_i*sk, so it cancels the message from both blocks of ct at once --
// including the mu*sk block, which needs no knowledge of sk because the gadget
// already carries it in the a-column.
//
// The second step is where the noise goes. theta = -(1 + Y + ... + Y^(C*W-1))
// is dense (see [thetaQP]), so it carries the phase error e to theta*e,
// inflating its variance by ||theta||_2^2 = C*W. That is inherent: multiplying
// by theta is the prefix sum that spreads a party's commitment across its w
// slots, and a prefix sum of a random walk grows like the square root of its
// length.
//
// The derivation belongs to the encoding of a commitment, not to the weight
// material, so [Aggregate] runs it per party per election rather than once per
// weight update. It is public arithmetic on a public ciphertext either way.
func deriveEncoder(params CircuitParams, theta ringqp.Poly, ctW, dst *rgsw.Ciphertext) {
	p := params.RLWE
	levelQ, levelP := ctW.LevelQ(), ctW.LevelP()

	for i := range dst.Value {
		gct, src := dst.Value[i], ctW.Value[i]
		for j := range gct.Value {
			for k := range gct.Value[j] {
				for l := range gct.Value[j][k] {
					gct.Value[j][k][l].Copy(src.Value[j][k][l])
				}
			}
		}
	}

	// Subtract 1 by adding -1 times the gadget vector to both blocks, the same
	// path rgsw.Encryptor.Encrypt uses to add the message in the first place.
	ringQ := p.RingQ().AtLevel(levelQ)
	minusOne := ringQ.NewPoly()
	for j, s := range ringQ.SubRings {
		minusOne.Coeffs[j][0] = s.Modulus - 1
	}
	ringQ.NTT(minusOne, minusOne)
	ringQ.MForm(minusOne, minusOne)

	if err := rlwe.AddPolyTimesGadgetVectorToGadgetCiphertext(
		minusOne,
		[]rlwe.GadgetCiphertext{dst.Value[0], dst.Value[1]},
		*p.RingQP(),
		ringQ.NewPoly(),
	); err != nil {
		panic(err)
	}

	// Multiply through by theta. Every polynomial of both blocks is scaled: the
	// operation is plaintext multiplication of the whole RGSW ciphertext.
	ringQP := p.RingQP().AtLevel(levelQ, levelP)
	for _, gct := range dst.Value {
		for i := range gct.Value {
			for j := range gct.Value[i] {
				for k := range gct.Value[i][j] {
					ringQP.MulCoeffsMontgomery(gct.Value[i][j][k], theta, gct.Value[i][j][k])
				}
			}
		}
	}
}

// thetaQP builds theta = 2/(Y - 1), in the NTT and Montgomery domain over QP
// so it can be applied directly to an RGSW ciphertext. [Aggregate] builds it
// once per election, outside the per-party loop: it depends on the layout
// alone, not on the weight.
//
// Y = X^S generates the subring Z[Y]/(Y^(C*W) + 1), Y^(C*W) = X^N = -1. (Y - 1)
// is not invertible there -- its norm is 2 -- but 2/(Y - 1) is, because
// (Y - 1)*(1 + Y + ... + Y^(C*W-1)) = Y^(C*W) - 1 = -2. Hence
//
//	theta = -(1 + Y + ... + Y^(C*W-1)),
//
// which as a polynomial in X is -1 at every stride-th coefficient below N.
// theta*(Y^w - 1) = 2*(1 + ... + Y^(w-1)) for every w <= C*W, and the weights
// never exceed W.
func thetaQP(params CircuitParams, ringQP ringqp.Ring) ringqp.Poly {
	theta := ringQP.NewPoly()

	terms := ringQP.N() / params.Stride // C*W
	for j, s := range ringQP.RingQ.SubRings {
		for k := 0; k < terms; k++ {
			theta.Q.Coeffs[j][params.Stride*k] = s.Modulus - 1
		}
	}
	for j, s := range ringQP.RingP.SubRings {
		for k := 0; k < terms; k++ {
			theta.P.Coeffs[j][params.Stride*k] = s.Modulus - 1
		}
	}

	ringQP.NTT(theta, theta)
	ringQP.MForm(theta, theta)

	return theta
}
