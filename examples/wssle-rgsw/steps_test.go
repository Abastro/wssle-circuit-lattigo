package wsslergsw

import (
	"math"
	"math/big"
	"math/bits"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func TestRegister(t *testing.T) {
	params := SetupParams(8)
	sk, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)

	// Every fragment nonzero and distinct, so a misplaced one shows.
	p := Party{Weight: 3, Commitment: fragmentsOf(params, 12, 5, 7, 9)}
	weight := EncryptWeight(enc, params, p.Weight)
	reg := Register(enc, params, p)

	assertOneHot(t, "ctW", decryptRGSW(enc, dec, eval, params, weight.CtW), params.Stride*int(p.Weight), 1)

	// ctEcd holds 2*(1 + Y + ... + Y^(w-1)), derived from ctW by public arithmetic.
	wantEcd := make([]float64, params.RLWE.N())
	for k := uint64(0); k < p.Weight; k++ {
		wantEcd[params.Stride*int(k)] = 2
	}
	assertVec(t, "ctEcd", decryptRGSW(enc, dec, eval, params, weight.CtEcd), wantEcd)

	// The registered commitment is h(Z), its fragments offset by one at
	// X^(k*N/C); the spread over the weight comes later.
	wantCtH := make([]float64, params.RLWE.N())
	for k, f := range storedFragments(params, p.Commitment) {
		wantCtH[params.FragmentIndex(k)] = commitFloat(f)
	}
	assertVec(t, "ctH", DecodeCoeffs(params.RLWE, dec.DecryptNew(reg.CtH), params.Scale()), wantCtH)

	// Applying ctEcd reproduces Fig. 1 line 5's ct_H, doubled: h(Z) in each of
	// the party's slots Y^j.
	ctEcd := rlwe.NewCiphertext(params.RLWE, 1, reg.CtH.Level())
	encodeH(eval, weight, reg, ctEcd)
	assertVec(t, "encodeH", DecodeCoeffs(params.RLWE, dec.DecryptNew(ctEcd), params.EncodedScale()), refHVec(params, p))

	nonzero, val := findNonzero(t, decryptRGSW(enc, dec, eval, params, reg.CtR))
	if math.Abs(val-1) > 1e-6 {
		t.Errorf("ctR: coeff[%d] = %v, want 1", nonzero, val)
	}
	if nonzero%params.Stride != 0 {
		t.Errorf("ctR: nonzero coeff at %d not aligned to stride %d", nonzero, params.Stride)
	}
	if nonzero/params.Stride >= int(params.TotalWt) {
		t.Errorf("ctR: sampled r=%d out of range [0, totalWeight)", nonzero/params.Stride)
	}
}

// TestAggregate builds registrations for the same 5-party/weight-8 scenario
// as the HIENAA reference test, and checks [Aggregate]'s two-pass external
// product fold against an independent plain-Go mirror of the same
// computation ([refCombineEcd]/[refAggregate]).
func TestAggregate(t *testing.T) {
	params := SetupParams(8)

	// The HIENAA scenario's 4-bit commitments in fragment 0, and distinct
	// values in the others, so a fragment that strays into another's
	// coefficient shows.
	parties := []Party{
		{Weight: 1, Commitment: fragmentsOf(params, 9, 1, 2, 3)},
		{Weight: 2, Commitment: fragmentsOf(params, 10, 4, 5, 6)},
		{Weight: 1, Commitment: fragmentsOf(params, 11, 7, 8, 9)},
		{Weight: 3, Commitment: fragmentsOf(params, 12, 10, 11, 12)},
		{Weight: 1, Commitment: fragmentsOf(params, 13, 13, 14, 15)},
	}

	sk, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)

	weights := EncryptWeights(enc, params, parties)
	regs, leaves := registerAll(t, enc, dec, eval, params, parties)

	agg := Aggregate(eval, weights, regs)
	want := refAggregate(params.RLWE.N(), params.Stride, leaves)

	// 2*Delta, since Aggregate now folds in the 2/(Y-1) of deriveEncoder.
	assertVec(t, "agg", DecodeCoeffs(params.RLWE, dec.DecryptNew(agg), params.EncodedScale()), want.h)
}

// TestTrace checks the relative trace Tr_{R/Z[Z]} [Elect] applies, on a dense
// random message: every coefficient off the multiples of N/C annihilated, and
// those kept scaled by N/C -- or returned as they are, when the ciphertext is
// first multiplied by (N/C)^-1 mod Q, as [Elect] does. The message fills every
// coefficient, so a trace that left anything off Z[Z] would show.
func TestTrace(t *testing.T) {
	params := SetupParams(8)
	sk, pk, evk := SetupKeys(params)

	enc := rlwe.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rlwe.NewEvaluator(params.RLWE, evk)

	N := params.RLWE.N()
	gain := N / params.Fragments
	ringQ := params.RLWE.RingQ()

	coeffs := make([]uint64, N)
	for i := range coeffs {
		coeffs[i] = uint64(rand.Intn(16))
	}

	ct, err := enc.EncryptNew(EncodeCoeffs(params.RLWE, coeffs, params.Delta))
	if err != nil {
		t.Fatal(err)
	}

	elems := ExtractionGaloisElements(params)
	if len(elems) != bits.Len(uint(gain))-1 {
		t.Errorf("%d Galois elements, want log2(N/C) = %d", len(elems), bits.Len(uint(gain))-1)
	}

	traced, err := Trace(eval, ct, elems)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float64, N)
	for i := 0; i < N; i += gain {
		want[i] = float64(gain) * float64(coeffs[i])
	}
	assertVec(t, "trace", DecodeCoeffs(params.RLWE, dec.DecryptNew(traced), params.Scale()), want)

	inv := new(big.Int).ModInverse(big.NewInt(int64(gain)), ringQ.Modulus())
	scaled := ct.CopyNew()
	ringQ.MulScalarBigint(scaled.Value[0], inv, scaled.Value[0])
	ringQ.MulScalarBigint(scaled.Value[1], inv, scaled.Value[1])

	traced, err = Trace(eval, scaled, elems)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < N; i += gain {
		want[i] = float64(coeffs[i])
	}
	assertVec(t, "(N/C)^-1 * trace", DecodeCoeffs(params.RLWE, dec.DecryptNew(traced), params.Scale()), want)
}

// TestThresholdDecrypt checks the committee's decryption against decryption
// under the whole key, which no one holds in an election and the test does:
// the combined phase must be the true phase plus the committee's flooding,
// that flooding within its hard bound m*F, and actually applied at scale F.
func TestThresholdDecrypt(t *testing.T) {
	params := SetupParams(8)
	sk, pk, _ := SetupKeys(params)
	keyShares := ShareSecretKey(params, sk)

	ringQ := params.RLWE.RingQ()
	sum := ringQ.NewPoly()
	for _, ks := range keyShares {
		ringQ.Add(sum, ks.Value, sum)
	}
	if !sum.Equal(&sk.Value.Q) {
		t.Fatal("key shares do not sum to the key")
	}

	// A full-width commitment's fragments, at the scale Elect's output carries.
	want := topCommitment(params.CommitmentBits())
	coeffs := make([]uint64, params.RLWE.N())
	for k, f := range storedFragments(params, want) {
		coeffs[params.FragmentIndex(k)] = f.Uint64()
	}
	ct, err := rlwe.NewEncryptor(params.RLWE, pk).EncryptNew(EncodeCoeffs(params.RLWE, coeffs, params.ResultScale()))
	if err != nil {
		t.Fatal(err)
	}

	shares := make([]DecryptionShare, params.CommitteeSize)
	for j, ks := range keyShares {
		shares[j] = PartialDecrypt(params, ct, ks)
	}
	phases := combinedPhase(params, ct, shares)

	if got, err := DecodeFragments(params, FinalDecrypt(params, ct, shares)); err != nil || got.Cmp(want) != 0 {
		t.Errorf("DecodeFragments = %v (%v), want %v", got, err, want)
	}

	truth := centeredCoeffs(params.RLWE, rlwe.NewDecryptor(params.RLWE, sk).DecryptNew(ct))
	F := params.FloodBound()
	bound := new(big.Int).Mul(F, big.NewInt(int64(params.CommitteeSize)))
	largest := new(big.Int)
	for k, ph := range phases {
		flood := new(big.Int).Sub(ph, truth[params.FragmentIndex(k)])
		if new(big.Int).Abs(flood).Cmp(bound) > 0 {
			t.Errorf("coeff %d: flooding 2^%.2f exceeds its bound m*F = 2^%.2f", k, log2Abs(flood), log2Abs(bound))
		}
		if new(big.Int).Abs(flood).Cmp(largest) > 0 {
			largest.Abs(flood)
		}
	}
	// A sum of m uniforms on [-F, F] has standard deviation F*sqrt(m/3); all
	// C of them below F/4 would mean the flooding is not being applied.
	if largest.Cmp(new(big.Int).Rsh(F, 2)) < 0 {
		t.Errorf("flooding too small: largest 2^%.2f, F = 2^%.2f", log2Abs(largest), log2Abs(F))
	}
	t.Logf("flooding: largest 2^%.2f, F = 2^%d, bound m*F = 2^%.2f",
		log2Abs(largest), params.LogErrorBound+params.SmudgeBits(), log2Abs(bound))
}

// registerAll registers every party and, by decrypting each ct_R, recovers
// the randomness it sampled so the plain-Go reference model can predict the
// same outcome.
func registerAll(t *testing.T, enc *rgsw.Encryptor, dec *rlwe.Decryptor, eval *rgsw.Evaluator, params CircuitParams, parties []Party) ([]*Registration, []refNode) {
	t.Helper()

	stride := params.Stride

	regs := make([]*Registration, len(parties))
	leaves := make([]refNode, len(parties))
	for i, p := range parties {
		regs[i] = Register(enc, params, p)

		idx, val := findNonzero(t, decryptRGSW(enc, dec, eval, params, regs[i].CtR))
		if math.Abs(val-1) > 1e-6 || idx%stride != 0 {
			t.Fatalf("party %d: unexpected ctR encoding at %d = %v", i, idx, val)
		}
		leaves[i] = refNode{w: p.Weight, r: uint64(idx / stride), h: refHVec(params, p)}
	}
	return regs, leaves
}

// refNode mirrors a [Registration] as plain (unencrypted) data: weight,
// (unreduced) randomness, and the coefficient vector ct_H decrypts to.
type refNode struct {
	w uint64
	r uint64
	h []float64
}

// refHVec builds the plain ct_H coefficient vector for one [Party]: its stored
// fragment k, h_k + 1, at Y^j Z^k = X^(S*j + k*N/C) for every slot j below its
// weight.
func refHVec(params CircuitParams, p Party) []float64 {
	h := make([]float64, params.RLWE.N())
	frags := storedFragments(params, p.Commitment)
	for j := 0; j < int(p.Weight); j++ {
		for k, f := range frags {
			h[params.Stride*j+params.FragmentIndex(k)] = commitFloat(f)
		}
	}
	return h
}

// fragmentsOf builds the commitment whose fragments are frags, least
// significant first.
func fragmentsOf(params CircuitParams, frags ...int64) *big.Int {
	fs := make([]*big.Int, params.Fragments)
	for k := range fs {
		fs[k] = new(big.Int)
		if k < len(frags) {
			fs[k].SetInt64(frags[k])
		}
	}
	return JoinFragments(params, fs)
}

// refCombineEcd mirrors [Aggregate]'s first pass on plain data: shift the
// accumulator by next's weight in the negacyclic ring, then add next's own
// commitment vector unshifted.
func refCombineEcd(rank, stride int, acc, next refNode) refNode {
	h := shiftNegacyclic(acc.h, int(next.w)*stride, rank)
	for i, v := range next.h {
		h[i] += v
	}
	return refNode{w: acc.w + next.w, r: acc.r + next.r, h: h}
}

// refAggregate mirrors [Aggregate] on plain data, matching its two-pass
// structure exactly.
func refAggregate(rank, stride int, nodes []refNode) refNode {
	acc := refNode{h: make([]float64, rank)}
	for _, n := range nodes {
		acc = refCombineEcd(rank, stride, acc, n)
	}
	for _, n := range nodes {
		acc.h = shiftNegacyclic(acc.h, int(n.r)*stride, rank)
	}
	return acc
}

// shiftNegacyclic shifts h by shift positions in Z[X]/(X^rank+1): index i
// moves to (i+shift) mod 2*rank, with a sign flip whenever that wraps past
// rank (X^rank = -1).
func shiftNegacyclic(h []float64, shift, rank int) []float64 {
	out := make([]float64, rank)
	period := 2 * rank
	shift = ((shift % period) + period) % period
	for i, v := range h {
		if v == 0 {
			continue
		}
		idx := i + shift
		if idx >= period {
			idx -= period
		}
		if idx < rank {
			out[idx] += v
		} else {
			out[idx-rank] -= v
		}
	}
	return out
}

// decryptRGSW recovers the monomial an [rgsw.Ciphertext] encrypts by applying
// it to a fresh encryption of 1 -- an RGSW ciphertext cannot be decrypted
// directly, only applied to an RLWE one.
func decryptRGSW(enc *rgsw.Encryptor, dec *rlwe.Decryptor, eval *rgsw.Evaluator, params CircuitParams, gsw *rgsw.Ciphertext) []float64 {
	one := make([]uint64, params.RLWE.N())
	one[0] = 1

	ct, err := enc.EncryptNew(EncodeCoeffs(params.RLWE, one, params.Delta))
	if err != nil {
		panic(err)
	}

	eval.ExternalProduct(ct, gsw, ct)

	return DecodeCoeffs(params.RLWE, dec.DecryptNew(ct), params.Scale())
}

// findNonzero locates the single coefficient that rounds to nonzero, and
// reports the absolute precision (raw value vs. its rounding) over every
// coefficient, so the noise magnitude stays visible even when the rounded
// values are right.
func findNonzero(t *testing.T, coeffs []float64) (int, float64) {
	t.Helper()

	idx, val, maxErr := -1, 0.0, 0.0
	for i, v := range coeffs {
		rounded := math.Round(v)
		maxErr = max(maxErr, math.Abs(v-rounded))
		if rounded != 0 {
			if idx != -1 {
				t.Fatalf("multiple nonzero coeffs at %d and %d", idx, i)
			}
			idx, val = i, rounded
		}
	}
	if idx == -1 {
		t.Fatal("no nonzero coeff found")
	}
	t.Logf("precision = %.1f bits", -math.Log2(maxErr))
	return idx, val
}

func assertOneHot(t *testing.T, name string, coeffs []float64, want int, value float64) {
	t.Helper()
	wantVec := make([]float64, len(coeffs))
	wantVec[want] = value
	assertVec(t, name, coeffs, wantVec)
}

// assertVec checks raw decoded coefficients against the exact expected
// integers by rounding, but reports the absolute precision (|got-want| before
// rounding) so the noise magnitude is visible in passing and failing runs
// alike.
func assertVec(t *testing.T, name string, got, want []float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}

	maxErr := 0.0
	for i := range got {
		absErr := math.Abs(got[i] - want[i])
		maxErr = max(maxErr, absErr)
		if math.Round(got[i]) != want[i] {
			t.Errorf("%s: coeff[%d] = %v, want %v (precision %.1f bits)", name, i, got[i], want[i], -math.Log2(absErr))
		}
	}
	t.Logf("%s: precision = %.1f bits", name, -math.Log2(maxErr))
}

// maxAbs returns the residual of largest magnitude.
func maxAbs(vs []*big.Int) *big.Int {
	best := new(big.Int)
	for _, v := range vs {
		if new(big.Int).Abs(v).Cmp(new(big.Int).Abs(best)) > 0 {
			best = v
		}
	}
	return best
}

// log2Abs reports a residual's magnitude in bits, exactly, without routing it
// through a float64.
func log2Abs(v *big.Int) float64 {
	a := new(big.Int).Abs(v)
	if a.Sign() == 0 {
		return math.Inf(-1)
	}
	f, _ := new(big.Float).SetInt(a).Float64()
	return math.Log2(f)
}

// commitFloat is a fragment as a float64, for the float reference model. It is
// exact only below 2^53; the reference therefore runs on small fragments, or on
// party labels (see predictWinner), never on full-width ones.
func commitFloat(c *big.Int) float64 {
	f, _ := new(big.Float).SetInt(c).Float64()
	return f
}
