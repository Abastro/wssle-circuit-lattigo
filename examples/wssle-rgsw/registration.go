package wsslergsw

import (
	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// Party is one participant's private input: its stake weight and its
// leader-commitment value. Commitment stands in for H(x_i) of the protocol --
// kept a plain integer here, since the circuit only ever moves it around.
type Party struct {
	Weight     uint64
	Commitment uint64
}

// Registration is the triple a party publishes in the Registration phase
// (Fig. 1, Register): ct_W and ct_R RGSW-encrypted, since [Aggregate] uses
// them only as the hidden multiplier of an external product, and ct_H a plain
// RLWE ciphertext, which is only ever added or used as the external product's
// RLWE operand.
type Registration struct {
	CtW, CtR *rgsw.Ciphertext
	CtH      *rlwe.Ciphertext
}

// Register runs the Registration phase for a single party: RGSW-encrypts the
// weight monomial Y^w = X^{S*w} and a fresh randomness monomial Y^r for
// r sampled uniformly from Z_W, and RLWE-encrypts the commitment encoding
// h*(Y^w - 1)/(Y - 1), i.e. h in each of the first w slots.
func Register(enc *rgsw.Encryptor, params CircuitParams, p Party) *Registration {
	if p.Weight > params.TotalWt {
		panic("party weight exceeds the total weight")
	}

	rank, stride := params.RLWE.N(), params.Stride

	// W is a power of two, so this reduction is unbiased.
	r := sampling.RandUint64() % params.TotalWt

	wCoeffs := make([]uint64, rank)
	wCoeffs[stride*int(p.Weight)] = 1

	rCoeffs := make([]uint64, rank)
	rCoeffs[stride*int(r)] = 1

	hCoeffs := make([]uint64, rank)
	for k := uint64(0); k < p.Weight; k++ {
		hCoeffs[stride*int(k)] = p.Commitment
	}

	ctH, err := enc.EncryptNew(EncodeCoeffs(params.RLWE, hCoeffs, params.Delta))
	if err != nil {
		panic(err)
	}

	return &Registration{
		CtW: encryptRGSW(enc, params, wCoeffs),
		CtR: encryptRGSW(enc, params, rCoeffs),
		CtH: ctH,
	}
}

// encryptRGSW RGSW-encrypts a monomial. The plaintext carries no scaling
// factor: an RGSW ciphertext encrypts mu as mu*G, and mu is the multiplier of
// the external product, not a scaled message.
func encryptRGSW(enc *rgsw.Encryptor, params CircuitParams, coeffs []uint64) *rgsw.Ciphertext {
	ct := rgsw.NewCiphertext(params.RLWE, params.RLWE.MaxLevelQ(), params.RLWE.MaxLevelP(), 0)
	if err := enc.Encrypt(EncodeCoeffs(params.RLWE, coeffs, 1), ct); err != nil {
		panic(err)
	}
	return ct
}
