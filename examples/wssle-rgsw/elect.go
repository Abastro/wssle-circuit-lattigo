package wsslergsw

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// Elect runs the Election phase (Fig. 1, Finalize line 12) on the aggregate
// from [Aggregate], which encrypts every party's fragments rotated by the
// joint randomness: the winner's k-th fragment sits at X^k. Each fragment is
// extracted by the full trace of the ring and put back in place,
//
//	ct_out = sum_{k < C} X^k * Tr(N^-1 X^-k * agg),
//
// so that one decryption reveals the winner's C fragments at X^0 .. X^(C-1),
// and nothing else.
//
// The trace has to be the full one, over all N automorphisms. The relative
// trace over the subring Y = X^S annihilates only the subring's non-constant
// coefficients, and would leave the fragments off the Y-lattice -- every
// other party's -- in the output that decryption reveals. The full trace is
// also what makes the N^-1 legitimate: it is an exact Z-linear map with
// Tr(e) = N*e_0 for every e in R, error included, so pre-multiplying by
// N^-1 mod Q cancels its gain on both message and error, and the fragments
// come out at 2*Delta rather than 2*Delta*N. What the trace adds is its own
// key-switching error.
//
// There is no ciphertext-ciphertext multiplication: [Aggregate] has already
// folded in every party's randomness by external product, so there is no
// relinearization key anywhere in this variant.
func Elect(eval *rgsw.Evaluator, params CircuitParams, agg *rlwe.Ciphertext) *rlwe.Ciphertext {
	p := params.RLWE
	ringQ := p.RingQ().AtLevel(agg.Level())
	N := p.N()

	nInv := new(big.Int).ModInverse(big.NewInt(int64(N)), ringQ.Modulus())

	out := agg.CopyNew()
	out.Value[0].Zero()
	out.Value[1].Zero()

	shifted := agg.CopyNew()
	for k := 0; k < params.Fragments; k++ {
		// N^-1 X^-k as one plaintext multiplier.
		mulConst(ringQ, agg, monomialNTT(ringQ, -k, nInv), shifted)

		traced, err := Trace(&eval.Evaluator, shifted, 2*N)
		if err != nil {
			panic(err)
		}

		mulConst(ringQ, traced, monomialNTT(ringQ, k, big.NewInt(1)), traced)
		ringQ.Add(out.Value[0], traced.Value[0], out.Value[0])
		ringQ.Add(out.Value[1], traced.Value[1], out.Value[1])
	}
	return out
}

// monomialNTT returns c * X^e for -N < e < N, in the NTT and Montgomery domain,
// as a plaintext multiplier for an NTT-domain ciphertext. X^-e = -X^(N-e).
func monomialNTT(ringQ *ring.Ring, e int, c *big.Int) ring.Poly {
	N := ringQ.N()
	coeff := new(big.Int).Set(c)
	if e < 0 {
		e += N
		coeff.Neg(coeff)
	}

	m := ringQ.NewPoly()
	qi, r := new(big.Int), new(big.Int)
	for j, s := range ringQ.SubRings[:ringQ.Level()+1] { // AtLevel keeps every prime
		m.Coeffs[j][e] = r.Mod(coeff, qi.SetUint64(s.Modulus)).Uint64()
	}
	ringQ.NTT(m, m)
	ringQ.MForm(m, m)
	return m
}

// mulConst sets dst = ct * m for a plaintext m from [monomialNTT]. Both are in
// the NTT domain, so this is a pointwise product per polynomial.
func mulConst(ringQ *ring.Ring, ct *rlwe.Ciphertext, m ring.Poly, dst *rlwe.Ciphertext) {
	if !ct.IsNTT {
		panic("mulConst: ciphertext must be in the NTT domain")
	}
	ringQ.MulCoeffsMontgomery(ct.Value[0], m, dst.Value[0])
	ringQ.MulCoeffsMontgomery(ct.Value[1], m, dst.Value[1])
}

// DecodeFragments reads the elected commitment h* off the phases of [Elect]'s
// output at X^0 .. X^(C-1), as [CombineShares] returns them: it rounds each
// fragment off [CircuitParams.ResultScale] and joins them.
//
// Each fragment is taken in magnitude, per Fig. 1 line 16 (h* <- |h'|): the
// winner's slot comes back negated when the accumulated random shift carries
// it past Y^W = -1. Its fragments wrap together, since they lie within one
// stride, so they share that one sign.
//
// The rounding is done in integers: a fragment is wider than float64's 53-bit
// mantissa in parameter sets A and B.
func DecodeFragments(params CircuitParams, phases []*big.Int) *big.Int {
	scale := params.ResultScale()
	half := new(big.Int).Rsh(scale, 1)

	frags := make([]*big.Int, params.Fragments)
	for k := range frags {
		// floor((v + S/2) / S): Div is Euclidean, hence a floor for the
		// positive divisor, including when v is negative.
		q := new(big.Int).Add(phases[k], half)
		q.Div(q, scale)
		frags[k] = q.Abs(q)
	}
	return JoinFragments(params, frags)
}

// DecodeResult is [DecodeFragments] on a plaintext decrypted under the whole
// key, which only the tests hold; an election decrypts through
// [PartialDecrypt] and [CombineShares].
func DecodeResult(params CircuitParams, pt *rlwe.Plaintext) *big.Int {
	return DecodeFragments(params, centeredCoeffs(params.RLWE, pt)[:params.Fragments])
}
