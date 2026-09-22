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
// weight material, registrations and randomness. The commitments fill the top
// of the H-bit range, so the no-wrap ceiling is exercised where it binds -- a
// commitment above [CircuitParams.MaxCommitment] would make [Register] panic.
//
// No flooding is added: threshold decryption is out of scope here (a single
// secret key stands in for the committee). The flooding only consumes budget,
// deterministically, and TestFloodingBudget checks that budget.
func TestWSSLE(t *testing.T) {
	t.Run("5 parties", func(t *testing.T) {
		testWSSLE(t, SetupParams(8), []Party{
			{Weight: 1, Commitment: big.NewInt(250)},
			{Weight: 2, Commitment: big.NewInt(251)},
			{Weight: 1, Commitment: big.NewInt(252)},
			{Weight: 3, Commitment: big.NewInt(253)},
			{Weight: 1, Commitment: big.NewInt(254)},
		})
	})

	if testing.Short() {
		t.Skip("full elections at n = 2048 on every parameter set; skipped in -short mode")
	}
	for _, ps := range ParamSets {
		t.Run(ps.Name, func(t *testing.T) {
			params := ps.Params()
			if top := topCommitment(ps.CoeffBits); top.Cmp(params.MaxCommitment()) > 0 {
				t.Fatalf("set %s cannot carry %d-bit commitments: MaxCommitment has %d bits",
					ps.Name, ps.CoeffBits, params.MaxCommitment().BitLen())
			}
			for trial := range electionTrials {
				// Sets run one after another; a set's trials run in parallel.
				t.Run("trial"+strconv.Itoa(trial), func(t *testing.T) {
					t.Parallel()
					testWSSLE(t, params, topParties(electionParties, ps.CoeffBits, ps.TotalWeight))
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
	pt := dec.DecryptNew(Elect(eval, agg, params.TotalWt)) // stand-in for ThFHE.Dec(ct_out, I)

	winner := predictWinner(t, params, leaves)
	want := parties[winner].Commitment

	// Every coefficient, exactly: the winner at coefficient 0, up to the sign
	// of the negacyclic wraparound (Fig. 1 line 16: h* <- W^-1 |h'|), and zero
	// everywhere the trace has annihilated.
	rounded := RoundCoeffs(params.RLWE, pt, params.ResultScale())
	if got := new(big.Int).Abs(rounded[0]); got.Cmp(want) != 0 {
		t.Errorf("coeff[0] = %v, want +-%v (party %d)", rounded[0], want, winner)
	}
	for i, v := range rounded[1:] {
		if v.Sign() != 0 {
			t.Errorf("coeff[%d] = %v, want 0", i+1, v)
		}
	}
	if got := DecodeResult(params, pt); got.Cmp(want) != 0 {
		t.Errorf("DecodeResult = %v, want %v (party %d)", got, want, winner)
	}

	// The noise is exact too; coefficient 0 is the one the trace amplifies by
	// W. The margin is how far it sits below the S/2 that rounding tolerates.
	res := Residuals(params.RLWE, pt, params.ResultScale())
	e0 := log2Abs(res[0])
	t.Logf("elected party %d of %d: |e_0| = 2^%.2f, max over the rest 2^%.2f, S/2 margin %.1f bits",
		winner, len(parties), e0, log2Abs(maxAbs(res[1:])), float64(params.ResultScale().BitLen()-2)-e0)
}

// predictWinner runs the plain-Go reference of [Aggregate] on the same weights
// and randomness as the real run, with each party's commitment replaced by its
// label i+1. The labels are small, so the float64 reference carries them
// exactly, whatever the width of the real commitments; the trace keeps only
// coefficient 0, so that is where the winner's label lands.
func predictWinner(t *testing.T, params CircuitParams, leaves []refNode) int {
	t.Helper()

	rank, stride := params.RLWE.N(), params.Stride
	labelled := make([]refNode, len(leaves))
	for i, l := range leaves {
		label := Party{Weight: l.w, Commitment: big.NewInt(int64(i + 1))}
		labelled[i] = refNode{w: l.w, r: l.r, h: refHVec(rank, stride, label)}
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
