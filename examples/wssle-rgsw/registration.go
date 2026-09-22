package wsslergsw

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// Party is one participant's private input: its stake weight and its
// leader-commitment value. Commitment stands in for H(x_i) of the protocol --
// a plain non-negative integer, since the circuit only ever moves it around,
// but a [big.Int], since the parameter sets carry 128 bits of it.
//
// Only Commitment is registered per election; Weight goes through
// [EncryptWeight] once, as a public parameter.
type Party struct {
	Weight     uint64
	Commitment *big.Int
}

// Registration is the pair a party publishes in the Registration phase:
// its commitment as a plain RLWE ciphertext, and its randomness RGSW-encrypted
// since [Aggregate] uses it as the hidden multiplier of an external product.
//
// Note the commitment is encrypted as Delta*(h_0 + h_1 X + ... ), its fragments
// as a polynomial of degree below C, not as the weight-dependent encoding
// h*(Y^w - 1)/(Y - 1) of Fig. 1 line 5. The encoding
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
//
// It rejects a commitment wider than C*H bits, and one with a fragment above
// [CircuitParams.MaxFragment]: the circuit would carry that fragment
// faithfully until decryption, where it wraps Q and decodes to another value.
func Register(enc *rgsw.Encryptor, params CircuitParams, p Party) *Registration {
	maxFrag := params.MaxFragment()
	for _, f := range SplitCommitment(params, p.Commitment) {
		if f.Cmp(maxFrag) > 0 {
			panic("commitment fragment above MaxFragment")
		}
	}

	// W is a power of two, so this reduction is unbiased.
	r := sampling.RandUint64() % params.TotalWt

	ctH, err := enc.EncryptNew(EncodeCommitment(params, p.Commitment))
	if err != nil {
		panic(err)
	}

	ctR := rgsw.NewCiphertext(params.RLWE, params.RLWE.MaxLevelQ(), params.RLWE.MaxLevelP(), 0)
	if err := enc.Encrypt(EncodeMonomial(params, r), ctR); err != nil {
		panic(err)
	}

	return &Registration{CtH: ctH, CtR: ctR}
}
