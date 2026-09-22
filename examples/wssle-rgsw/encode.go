package wsslergsw

import (
	"math/big"
	"math/bits"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// EncodeCoeffs builds a coefficient-encoded plaintext holding delta*coeffs,
// in the NTT domain.
//
// With [EncodeMonomial] and [EncodeCommitment], this is all the encoding the
// circuit needs. The scheme packages encode
// through the canonical embedding (slots), whereas every value here lives in
// a coefficient of the ring: a weight is the monomial X^{S*w}, a commitment
// block is a run of equal coefficients at stride S. delta is 1 for the RGSW
// operands -- an RGSW ciphertext encrypts a raw ring element used as a
// multiplier, not a scaled message -- and [CircuitParams.Delta] for a scaled
// message. ct_H itself goes through [EncodeCommitment], since a commitment
// need not fit a uint64.
func EncodeCoeffs(params rlwe.Parameters, coeffs []uint64, delta *big.Int) *rlwe.Plaintext {
	pt := rlwe.NewPlaintext(params, params.MaxLevelQ())

	ringQ := params.RingQ().AtLevel(pt.Level())
	qi, dq := new(big.Int), new(big.Int)
	for j, s := range ringQ.SubRings {
		q, dst := s.Modulus, pt.Value.Coeffs[j]
		d := dq.Mod(delta, qi.SetUint64(q)).Uint64()
		for i, c := range coeffs {
			dst[i] = mulMod(c, d, q)
		}
	}

	ringQ.NTT(pt.Value, pt.Value)
	pt.IsNTT = true
	pt.IsMontgomery = false

	return pt
}

// EncodeCommitment builds the plaintext ct_h encrypts: Delta times the
// commitment as the subring element
//
//	h(Z) = sum_k (h_k + 1) Z^k,   Z = X^(N/C),   h = sum_k h_k 2^(k*H),
//
// fragment k at X^(k*N/C) ([CircuitParams.FragmentIndex]), in the NTT domain.
// Each fragment is offset by one so that it is strictly positive, which is what
// lets [DecodeFragments] read the winner's rotation off the signs. [encodeH]
// then places h(Z) in every slot Y^j of the party's weight.
func EncodeCommitment(params CircuitParams, h *big.Int) *rlwe.Plaintext {
	pt := rlwe.NewPlaintext(params.RLWE, params.RLWE.MaxLevelQ())

	ringQ := params.RLWE.RingQ().AtLevel(pt.Level())
	qi, v := new(big.Int), new(big.Int)
	for k, frag := range SplitCommitment(params, h) {
		v.Add(frag, big.NewInt(1))
		v.Mul(v, params.Delta)
		for j, s := range ringQ.SubRings {
			pt.Value.Coeffs[j][params.FragmentIndex(k)] = new(big.Int).Mod(v, qi.SetUint64(s.Modulus)).Uint64()
		}
	}

	ringQ.NTT(pt.Value, pt.Value)
	pt.IsNTT = true
	pt.IsMontgomery = false

	return pt
}

// SplitCommitment returns the C fragments of h, least significant first, each
// FragmentBits wide. h must be non-negative and fit CommitmentBits.
func SplitCommitment(params CircuitParams, h *big.Int) []*big.Int {
	if h.Sign() < 0 || uint(h.BitLen()) > params.CommitmentBits() {
		panic("commitment outside [0, 2^(C*H))")
	}
	mask := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), params.FragmentBits), big.NewInt(1))
	frags := make([]*big.Int, params.Fragments)
	for k := range frags {
		frags[k] = new(big.Int).Rsh(h, uint(k)*params.FragmentBits)
		frags[k].And(frags[k], mask)
	}
	return frags
}

// JoinFragments is the inverse of [SplitCommitment].
func JoinFragments(params CircuitParams, frags []*big.Int) *big.Int {
	h := new(big.Int)
	for k := len(frags) - 1; k >= 0; k-- {
		h.Lsh(h, params.FragmentBits)
		h.Add(h, frags[k])
	}
	return h
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
// plaintext's coefficients and divides them by divisor.
//
// divisor is [CircuitParams.Scale] for an ordinary plaintext and
// [CircuitParams.ResultScale] for [Elect]'s output.
//
// The float64 result is for inspection only. It cannot round off a commitment
// wider than 53 bits -- [RoundCoeffs] does that, in integers -- nor measure
// noise: once the scale is large the message dominates its own
// coefficient, and a value near 2^94 has a float64 ulp of 2^42. Use
// [Residuals] for that.
func DecodeCoeffs(params rlwe.Parameters, pt *rlwe.Plaintext, divisor *big.Int) []float64 {
	centered := centeredCoeffs(params, pt)

	div := new(big.Float).SetInt(divisor)
	out := make([]float64, len(centered))
	for i, v := range centered {
		out[i], _ = new(big.Float).Quo(new(big.Float).SetInt(v), div).Float64()
	}
	return out
}

// RoundCoeffs is the exact counterpart of [DecodeCoeffs]: each coefficient,
// centre-lifted, divided by scale and rounded to the nearest integer, in
// integer arithmetic throughout.
func RoundCoeffs(params rlwe.Parameters, pt *rlwe.Plaintext, scale *big.Int) []*big.Int {
	half := new(big.Int).Rsh(scale, 1)

	out := centeredCoeffs(params, pt)
	for _, v := range out {
		// floor((v + scale/2) / scale): Div is Euclidean, hence a floor for
		// the positive divisor, including when v is negative.
		v.Add(v, half)
		v.Div(v, scale)
	}
	return out
}

// Residuals returns each coefficient's exact signed distance to the nearest
// multiple of scale -- that is, the noise it carries -- in [-scale/2, scale/2).
//
// This is the measurement [DecodeCoeffs] cannot make: it never forms the
// message and the noise as one float, so the constant coefficient, which the
// trace amplifies by W and which therefore carries the largest noise in the
// ciphertext, stays visible.
func Residuals(params rlwe.Parameters, pt *rlwe.Plaintext, scale *big.Int) []*big.Int {
	half := new(big.Int).Rsh(scale, 1)

	out := centeredCoeffs(params, pt)
	for _, v := range out {
		v.Add(v, half)
		v.Mod(v, scale) // Mod is Euclidean, so the result is non-negative
		v.Sub(v, half)
	}
	return out
}

// centeredCoeffs lifts pt's coefficients to the centred range of Q.
func centeredCoeffs(params rlwe.Parameters, pt *rlwe.Plaintext) []*big.Int {
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
	return centered
}

// mulMod returns a*b mod q for any a, b < 2^64, via the 128-bit product, so
// that encoding cannot silently overflow as the commitment size grows.
func mulMod(a, b, q uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	_, r := bits.Div64(hi%q, lo, q)
	return r
}
