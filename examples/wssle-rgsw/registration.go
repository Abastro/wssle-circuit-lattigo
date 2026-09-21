package wsslergsw

import (
	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// Party is one participant's private input: its stake weight and its
// leader-commitment value. Commitment stands in for H(x_i) of the protocol --
// kept a plain integer here, since the circuit only ever moves it around.
//
// Only Commitment is registered per election; Weight goes through
// [EncryptWeight] once, as a public parameter.
type Party struct {
	Weight     uint64
	Commitment uint64
}

// Registration is the pair a party publishes in the Registration phase:
// its commitment as a plain RLWE ciphertext, and its randomness RGSW-encrypted
// since [Aggregate] uses it as the hidden multiplier of an external product.
//
// Note the commitment is encrypted as the *constant* Delta*h, not as the
// weight-dependent encoding h*(Y^w - 1)/(Y - 1) of Fig. 1 line 5. The encoding
// is applied by the aggregator instead, from the public [Weight.CtEcd] (see
// [encodeH]), so a party cannot register a commitment spread over a weight
// other than its own.
type Registration struct {
	CtH *rlwe.Ciphertext
	CtR *rgsw.Ciphertext
}

// Register runs the Registration phase for a single party: RLWE-encrypts the
// commitment and RGSW-encrypts a fresh randomness monomial Y^r for r sampled
// uniformly from Z_W.
func Register(enc *rgsw.Encryptor, params CircuitParams, p Party) *Registration {
	// W is a power of two, so this reduction is unbiased.
	r := sampling.RandUint64() % params.TotalWt

	hCoeffs := make([]uint64, params.RLWE.N())
	hCoeffs[0] = p.Commitment

	ctH, err := enc.EncryptNew(EncodeCoeffs(params.RLWE, hCoeffs, params.Delta))
	if err != nil {
		panic(err)
	}

	ctR := rgsw.NewCiphertext(params.RLWE, params.RLWE.MaxLevelQ(), params.RLWE.MaxLevelP(), 0)
	if err := enc.Encrypt(EncodeMonomial(params, r), ctR); err != nil {
		panic(err)
	}

	return &Registration{CtH: ctH, CtR: ctR}
}
