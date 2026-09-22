package wsslergsw

import (
	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
)

// Weight is the public encryption of one party's stake. Stake is a public
// parameter rather than a per-election input (Fig. 1, Setup line 3:
// pp <- pp u {Enc(w_i)}), so both ciphertexts are produced once per weight
// update and reused across elections.
type Weight struct {
	// CtW encrypts Y^w, the shift [Aggregate]'s first pass applies.
	CtW *rgsw.Ciphertext
	// CtEcd encrypts 2*(Y^w - 1)/(Y - 1) = 2*(1 + Y + ... + Y^(w-1)), which
	// turns a party's Enc(h) into its encoded contribution (see [encodeH]).
	// It is derived from CtW by [deriveEncoder], not encrypted separately, so
	// that the two agree on w by construction rather than by proof.
	CtEcd *rgsw.Ciphertext
}

// EncryptWeight builds the [Weight] material for a single stake value.
func EncryptWeight(enc *rgsw.Encryptor, params CircuitParams, weight uint64) *Weight {
	if weight > params.TotalWt {
		panic("party weight exceeds the total weight")
	}

	ctW := rgsw.NewCiphertext(params.RLWE, params.RLWE.MaxLevelQ(), params.RLWE.MaxLevelP(), 0)
	if err := enc.Encrypt(EncodeMonomial(params, weight), ctW); err != nil {
		panic(err)
	}

	return &Weight{CtW: ctW, CtEcd: deriveEncoder(params, ctW)}
}

// EncryptWeights runs [EncryptWeight] over a whole party set.
func EncryptWeights(enc *rgsw.Encryptor, params CircuitParams, parties []Party) []*Weight {
	weights := make([]*Weight, len(parties))
	for i, p := range parties {
		weights[i] = EncryptWeight(enc, params, p.Weight)
	}
	return weights
}

// deriveEncoder turns RGSW(Y^w) into RGSW(2*(Y^w - 1)/(Y - 1)) by public
// arithmetic alone, in two steps:
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
func deriveEncoder(params CircuitParams, ctW *rgsw.Ciphertext) *rgsw.Ciphertext {
	p := params.RLWE
	levelQ, levelP := ctW.LevelQ(), ctW.LevelP()

	out := &rgsw.Ciphertext{Value: [2]rlwe.GadgetCiphertext{
		*ctW.Value[0].CopyNew(),
		*ctW.Value[1].CopyNew(),
	}}

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
		[]rlwe.GadgetCiphertext{out.Value[0], out.Value[1]},
		*p.RingQP(),
		ringQ.NewPoly(),
	); err != nil {
		panic(err)
	}

	// Multiply through by theta. Every polynomial of both blocks is scaled: the
	// operation is plaintext multiplication of the whole RGSW ciphertext.
	ringQP := p.RingQP().AtLevel(levelQ, levelP)
	theta := thetaQP(params, ringQP)
	for _, gct := range out.Value {
		for i := range gct.Value {
			for j := range gct.Value[i] {
				for k := range gct.Value[i][j] {
					ringQP.MulCoeffsMontgomery(gct.Value[i][j][k], theta, gct.Value[i][j][k])
				}
			}
		}
	}

	return out
}

// thetaQP builds theta = 2/(Y - 1), in the NTT and Montgomery domain over QP
// so it can be applied directly to an RGSW ciphertext.
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
