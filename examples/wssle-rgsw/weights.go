package wsslergsw

import (
	"github.com/tuneinsight/lattigo/v6/core/rgsw"
)

// EncryptWeight encrypts one party's stake as RGSW(Y^w), the shift
// [Aggregate]'s first pass applies. Stake is a public parameter rather than a
// per-election input (Fig. 1, Setup line 3: pp <- pp u {Enc(w_i)}), so it is
// produced once per weight update and reused across elections.
//
// The encoder 2*(Y^w - 1)/(Y - 1) is not part of this material: it is what
// turns a registered commitment into its encoded contribution, so [Aggregate]
// derives it from Enc(Y^w) as it builds that contribution (see
// [deriveEncoder]).
func EncryptWeight(enc *rgsw.Encryptor, params CircuitParams, weight uint64) *rgsw.Ciphertext {
	if weight > params.TotalWt {
		panic("party weight exceeds the total weight")
	}

	ctW := rgsw.NewCiphertext(params.RLWE, params.RLWE.MaxLevelQ(), params.RLWE.MaxLevelP(), 0)
	if err := enc.Encrypt(EncodeMonomial(params, weight), ctW); err != nil {
		panic(err)
	}

	return ctW
}

// EncryptWeights runs [EncryptWeight] over a whole party set.
func EncryptWeights(enc *rgsw.Encryptor, params CircuitParams, parties []Party) []*rgsw.Ciphertext {
	weights := make([]*rgsw.Ciphertext, len(parties))
	for i, p := range parties {
		weights[i] = EncryptWeight(enc, params, p.Weight)
	}
	return weights
}
