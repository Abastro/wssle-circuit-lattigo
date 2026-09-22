package wsslergsw

import (
	"math"
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
// and fill the top of the range, so every fragment sits near 2^H - 1 and the
// no-wrap ceiling is exercised where it binds -- a fragment above
// [CircuitParams.MaxFragment] would make [Register] panic.
//
// No flooding is added: threshold decryption is out of scope here (a single
// secret key stands in for the committee). The flooding only consumes budget,
// deterministically, and TestFloodingBudget checks that budget.
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
			if top := topCommitment(ps.FragmentBits); top.Cmp(params.MaxFragment()) > 0 {
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

func testWSSLE(t *testing.T, params CircuitParams, parties []Party) {
	sk, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)

	weights := EncryptWeights(enc, params, parties)
	regs, leaves := registerAll(t, enc, dec, eval, params, parties)

	agg := Aggregate(eval, Identity(enc.Encryptor, params), weights, regs)
	pt := dec.DecryptNew(Elect(eval, params, agg)) // stand-in for ThFHE.Dec(ct_out, I)

	winner := predictWinner(t, params, leaves)
	want := parties[winner].Commitment

	// Every coefficient, exactly: the winner's fragments at X^0 .. X^(C-1),
	// sharing one sign (the negacyclic wraparound of Fig. 1 line 16,
	// h* <- |h'|), and zero everywhere the traces have annihilated.
	rounded := RoundCoeffs(params.RLWE, pt, params.ResultScale())
	sign := 0
	for k, f := range SplitCommitment(params, want) {
		got := rounded[k]
		if new(big.Int).Abs(got).Cmp(f) != 0 {
			t.Errorf("fragment %d = %v, want +-%v (party %d)", k, got, f, winner)
		}
		if s := got.Sign(); s != 0 {
			if sign != 0 && s != sign {
				t.Errorf("fragment %d has sign %d, fragment 0.. have %d", k, s, sign)
			}
			sign = s
		}
	}
	for i, v := range rounded[params.Fragments:] {
		if v.Sign() != 0 {
			t.Errorf("coeff[%d] = %v, want 0", params.Fragments+i, v)
		}
	}
	if got := DecodeResult(params, pt); got.Cmp(want) != 0 {
		t.Errorf("DecodeResult = %v, want %v (party %d)", got, want, winner)
	}

	// The noise is exact too. The margin is how far the largest fragment error
	// sits below the S/2 that rounding tolerates.
	res := Residuals(params.RLWE, pt, params.ResultScale())
	eFrag := log2Abs(maxAbs(res[:params.Fragments]))
	t.Logf("elected party %d of %d: max fragment error 2^%.2f, max over the rest 2^%.2f, S/2 margin %.1f bits",
		winner, len(parties), eFrag, log2Abs(maxAbs(res[params.Fragments:])), float64(params.ResultScale().BitLen()-2)-eFrag)
}

// predictWinner runs the plain-Go reference of [Aggregate] on the same weights
// and randomness as the real run, with each party's commitment replaced by its
// label i+1, a single small fragment. The float64 reference carries the labels
// exactly, whatever the width of the real commitments, and the winner's lands
// at coefficient 0, where its fragment 0 does.
func predictWinner(t *testing.T, params CircuitParams, leaves []refNode) int {
	t.Helper()

	rank, stride := params.RLWE.N(), params.Stride
	labelled := make([]refNode, len(leaves))
	for i, l := range leaves {
		label := Party{Weight: l.w, Commitment: big.NewInt(int64(i + 1))}
		labelled[i] = refNode{w: l.w, r: l.r, h: refHVec(params, label)}
	}

	h0 := math.Abs(refAggregate(rank, stride, labelled).h[0])
	winner := int(h0) - 1
	if float64(winner+1) != h0 || winner < 0 || winner >= len(leaves) {
		t.Fatalf("reference put %v at coefficient 0, not a party label", h0)
	}
	return winner
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
