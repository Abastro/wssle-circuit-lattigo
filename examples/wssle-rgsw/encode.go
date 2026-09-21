package wsslergsw

import (
	"math/big"
	"math/bits"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// EncodeCoeffs builds a coefficient-encoded plaintext holding delta*coeffs,
// in the NTT domain.
//
// This is the whole "encoder" this circuit needs. The scheme packages encode
// through the canonical embedding (slots), whereas every value here lives in
// a coefficient of the ring: a weight is the monomial X^{S*w}, a commitment
// block is a run of equal coefficients at stride S. delta is 1 for the RGSW
// operands -- an RGSW ciphertext encrypts a raw ring element used as a
// multiplier, not a scaled message -- and [CircuitParams.Delta] for ct_H.
func EncodeCoeffs(params rlwe.Parameters, coeffs []uint64, delta uint64) *rlwe.Plaintext {
	pt := rlwe.NewPlaintext(params, params.MaxLevelQ())

	ringQ := params.RingQ().AtLevel(pt.Level())
	for j, s := range ringQ.SubRings {
		q, dst := s.Modulus, pt.Value.Coeffs[j]
		for i, c := range coeffs {
			dst[i] = mulMod(c, delta, q)
		}
	}

	ringQ.NTT(pt.Value, pt.Value)
	pt.IsNTT = true
	pt.IsMontgomery = false

	return pt
}

// EncodeMonomial builds the plaintext Y^exp = X^{Stride*exp}, for any
// 0 <= exp <= W. It carries no scaling factor: the monomials of this circuit
// are all RGSW operands.
//
// exp == W is the one case that wraps: Y^W = X^N = -1, which lands on the
// constant coefficient with a sign flip rather than off the end of the ring.
// That is a party holding the entire stake, which [Register] must not reject.
func EncodeMonomial(params CircuitParams, exp uint64) *rlwe.Plaintext {
	pt := rlwe.NewPlaintext(params.RLWE, params.RLWE.MaxLevelQ())

	ringQ := params.RLWE.RingQ().AtLevel(pt.Level())

	idx, negated := params.Stride*int(exp), false
	if idx >= ringQ.N() {
		idx, negated = idx-ringQ.N(), true
	}

	for j, s := range ringQ.SubRings {
		if negated {
			pt.Value.Coeffs[j][idx] = s.Modulus - 1
		} else {
			pt.Value.Coeffs[j][idx] = 1
		}
	}

	ringQ.NTT(pt.Value, pt.Value)
	pt.IsNTT = true
	pt.IsMontgomery = false

	return pt
}

// DecodeCoeffs is the inverse of [EncodeCoeffs]: it centre-lifts the
// plaintext's coefficients and divides them by divisor, returning the raw
// (unrounded) values so callers can report the noise magnitude.
//
// divisor is delta for an ordinary plaintext, and delta*W for [Elect]'s
// output, whose coefficients carry the trace's factor of W (Fig. 1, line 16:
// h* <- W^-1 |h'|). Undoing W here rather than homomorphically is exact and
// free: the election result is public, so the division happens in the clear
// on an integer.
func DecodeCoeffs(params rlwe.Parameters, pt *rlwe.Plaintext, divisor uint64) []float64 {
	ringQ := params.RingQ().AtLevel(pt.Level())

	p := ringQ.NewPoly()
	if pt.IsNTT {
		ringQ.INTT(pt.Value, p)
	} else {
		pt.Value.CopyLvl(pt.Level(), p)
	}

	centered := make([]*big.Int, ringQ.N())
	for i := range centered {
		centered[i] = new(big.Int)
	}
	ringQ.PolyToBigintCentered(p, 1, centered)

	div := new(big.Float).SetUint64(divisor)
	out := make([]float64, len(centered))
	for i, v := range centered {
		out[i], _ = new(big.Float).Quo(new(big.Float).SetInt(v), div).Float64()
	}
	return out
}

// mulMod returns a*b mod q for any a, b < 2^64, via the 128-bit product, so
// that encoding cannot silently overflow as the commitment size grows.
func mulMod(a, b, q uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	_, r := bits.Div64(hi%q, lo, q)
	return r
}
