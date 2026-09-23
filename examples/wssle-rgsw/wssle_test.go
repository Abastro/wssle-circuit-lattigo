package wsslergsw

import (
	"math/big"
	"strconv"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

const (
	electionTrials  = 10   // independent elections per parameter set
	electionParties = 2048 // the largest election size benchmarked
)

// TestWSSLE runs the full circuit end-to-end and checks the elected
// commitment exactly against an independent prediction.
//
// Each parameter set runs electionTrials elections, every one with fresh keys,
// weight material, registrations and randomness. The commitments are 128-bit
// and fill the top of the range, so every stored fragment sits at 2^H and the
// no-wrap ceiling is exercised where it binds -- a fragment above
// [CircuitParams.MaxFragment] would make [Register] panic.
//
// The result is decrypted by the whole committee, flooding included, and
// checked again under the whole key, which only the test holds.
func TestWSSLE(t *testing.T) {
	t.Run("5 parties", func(t *testing.T) {
		params := SetupParams(8)
		parties := topParties(5, params.CommitmentBits(), 5)
		for i, w := range []uint64{1, 2, 1, 3, 1} {
			parties[i].Weight = w
		}
		testWSSLE(t, params, parties)
	})

	if testing.Short() {
		t.Skip("full elections at n = 2048 on every parameter set; skipped in -short mode")
	}
	for _, ps := range ParamSets {
		t.Run(ps.Name, func(t *testing.T) {
			params := ps.Params()
			// Stored fragments, h_k + 1, reach 2^H.
			if top := new(big.Int).Lsh(big.NewInt(1), ps.FragmentBits); top.Cmp(params.MaxFragment()) > 0 {
				t.Fatalf("set %s cannot carry %d-bit fragments: MaxFragment has %d bits",
					ps.Name, ps.FragmentBits, params.MaxFragment().BitLen())
			}
			for trial := range electionTrials {
				// Sets run one after another; a set's trials run in parallel.
				t.Run("trial"+strconv.Itoa(trial), func(t *testing.T) {
					t.Parallel()
					testWSSLE(t, params, topParties(electionParties, params.CommitmentBits(), ps.TotalWeight))
				})
			}
		})
	}
}

// TestWSSLESweep runs the elections the benchmark times: every parameter set
// at n = 2, 4, .., 2048 parties sharing the set's W equally, plus n = 1, a
// single party holding the whole stake -- the one case where [EncodeMonomial]
// is asked for Y^W itself, which is Z. Each (set, n) runs sweepTrials elections, all in parallel
// within a set; each is checked exactly as in TestWSSLE, and logs the rotation
// Z^c the winner's fragments came back with, so the wraparound is on record.
func TestWSSLESweep(t *testing.T) {
	if testing.Short() {
		t.Skip("full elections over the whole benchmark grid; skipped in -short mode")
	}
	const sweepTrials = 3
	for _, ps := range ParamSets {
		t.Run(ps.Name, func(t *testing.T) {
			params := ps.Params()
			for n := 1; n <= 2048; n *= 2 {
				for trial := range sweepTrials {
					name := strconv.Itoa(n) + "_parties/trial" + strconv.Itoa(trial)
					t.Run(name, func(t *testing.T) {
						t.Parallel()
						testWSSLE(t, params, topParties(n, params.CommitmentBits(), ps.TotalWeight))
					})
				}
			}
		})
	}
}

func testWSSLE(t *testing.T, params CircuitParams, parties []Party) {
	sk, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)

	weights := EncryptWeights(enc, params, parties)
	regs, leaves := registerAll(t, enc, dec, eval, params, parties)

	agg := Aggregate(eval, params, weights, regs)
	ctOut := Elect(eval, params, agg)

	winner, rotation := predictWinner(params, parties, leaves)
	want := parties[winner].Commitment

	// The election result, as the committee decrypts it: every member's
	// flooded share of the C output coefficients, combined and rounded.
	keyShares := ShareSecretKey(params, sk)
	shares := make([]DecryptionShare, len(keyShares))
	for j, ks := range keyShares {
		shares[j] = PartialDecrypt(params, ctOut, ks)
	}
	phases := combinedPhase(params, ctOut, shares)
	if got, err := DecodeFragments(params, FinalDecrypt(params, ctOut, shares)); err != nil || got.Cmp(want) != 0 {
		t.Errorf("committee decrypted %v (%v), want %v (party %d)", got, err, want, winner)
	}

	// Under the whole key, which only the test holds: every coefficient,
	// exactly -- the winner's stored fragments h_k + 1 rotated by Z^c at
	// X^(k*N/C), and zero everywhere the trace has annihilated, so nothing
	// else is revealed.
	pt := dec.DecryptNew(ctOut)
	rounded := RoundCoeffs(params.RLWE, pt, params.ResultScale())
	isFragment := make(map[int]bool)
	for k, v := range rotateZ(storedFragments(params, want), rotation) {
		isFragment[params.FragmentIndex(k)] = true
		if got := rounded[params.FragmentIndex(k)]; got.Cmp(v) != 0 {
			t.Errorf("coefficient of Z^%d = %v, want %v (party %d, c = %d)", k, got, v, winner, rotation)
		}
	}
	for i, v := range rounded {
		if !isFragment[i] && v.Sign() != 0 {
			t.Errorf("coeff[%d] = %v, want 0", i, v)
		}
	}
	if got, err := DecodeResult(params, pt); err != nil || got.Cmp(want) != 0 {
		t.Errorf("DecodeResult = %v (%v), want %v (party %d)", got, err, want, winner)
	}

	// The errors are exact too: the evaluation error alone, under the whole
	// key, and with the committee's flooding on top, each against the S/2 that
	// rounding tolerates.
	scale := params.ResultScale()
	halfBits := float64(scale.BitLen() - 2)
	res := Residuals(params.RLWE, pt, scale)
	var fragRes []*big.Int
	for i, v := range res {
		if isFragment[i] {
			fragRes = append(fragRes, v)
		}
	}
	eFrag := log2Abs(maxAbs(fragRes))
	eTotal := log2Abs(maxAbs(residualsOf(phases, scale)))
	t.Logf("elected party %d of %d (weight %d, c = %d): eval error 2^%.2f (margin %.1f bits), with flooding 2^%.2f (margin %.1f bits)",
		winner, len(parties), parties[winner].Weight, rotation, eFrag, halfBits-eFrag, eTotal, halfBits-eTotal)
}

// residualsOf is each value's signed distance to the nearest multiple of scale.
func residualsOf(vs []*big.Int, scale *big.Int) []*big.Int {
	half := new(big.Int).Rsh(scale, 1)
	out := make([]*big.Int, len(vs))
	for i, v := range vs {
		r := new(big.Int).Add(v, half)
		r.Mod(r, scale)
		out[i] = r.Sub(r, half)
	}
	return out
}

// predictWinner computes, from the weights and the randomness the parties
// registered, which party the election must elect and the rotation Z^c its
// fragments must come back with (references/circuit, Correctness). The first
// pass leaves party i in the slots from sum_{l>i} w_l on, below
// sum_{l>=i} w_l; the second rotates everything by Y^r, r = sum_i r_i, which
// brings slot j* = -r mod W to Y^0 after crossing Y^W = Z
// c = floor((j* + r)/W) times, mod 2C since Z^(2C) = 1.
func predictWinner(params CircuitParams, parties []Party, leaves []refNode) (winner, rotation int) {
	W := int(params.TotalWt)
	r := 0
	for _, l := range leaves {
		r += int(l.r)
	}
	jStar := ((-r)%W + W) % W

	above := 0 // total weight of the parties after i
	for i := len(parties) - 1; i >= 0; i-- {
		w := int(parties[i].Weight)
		if jStar >= above && jStar < above+w {
			winner = i
		}
		above += w
	}
	return winner, ((jStar + r) / W) % (2 * params.Fragments)
}

// storedFragments is h's fragments as the circuit carries them, h_k + 1.
func storedFragments(params CircuitParams, h *big.Int) []*big.Int {
	frags := SplitCommitment(params, h)
	for _, f := range frags {
		f.Add(f, big.NewInt(1))
	}
	return frags
}

// rotateZ multiplies the subring element sum_k v_k Z^k by Z^c, Z^C = -1.
func rotateZ(v []*big.Int, c int) []*big.Int {
	C := len(v)
	out := make([]*big.Int, C)
	for k := range v {
		out[k] = new(big.Int).Set(v[k])
	}
	for ; c > 0; c-- {
		last := out[C-1]
		copy(out[1:], out[:C-1])
		out[0] = last.Neg(last)
	}
	return out
}

// topParties builds n parties sharing the total weight equally, with distinct
// commitments at the very top of the bits-wide range.
func topParties(n int, bits uint, totalWeight uint64) []Party {
	top := topCommitment(bits)
	parties := make([]Party, n)
	for i := range parties {
		c := new(big.Int).Sub(top, big.NewInt(int64(i)))
		parties[i] = Party{Weight: totalWeight / uint64(n), Commitment: c}
	}
	return parties
}

// topCommitment is 2^bits - 1, the largest bits-wide commitment.
func topCommitment(bits uint) *big.Int {
	return new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), bits), big.NewInt(1))
}

// uniformParties builds n weight-1 parties with distinct 32-bit commitments,
// so the total weight is n and every party owns exactly one slot.
func uniformParties(n int) []Party {
	parties := make([]Party, n)
	for i := range parties {
		parties[i] = Party{Weight: 1, Commitment: big.NewInt(int64(0xFFFFFFFF - i))}
	}
	return parties
}
