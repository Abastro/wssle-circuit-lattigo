package wsslergsw

import (
	"math"
	"strconv"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// TestWSSLE runs the full circuit end-to-end and checks the elected
// commitment against an independent prediction. [refAggregate] already bakes
// in the randomness shift (mirroring [Aggregate]'s two-pass structure), so
// the winning commitment is simply want.h[0] -- no separate slot
// reconstruction is needed.
func TestWSSLE(t *testing.T) {
	t.Run("5 parties", func(t *testing.T) {
		testWSSLE(t, []Party{
			{Weight: 1, Commitment: 250},
			{Weight: 2, Commitment: 251},
			{Weight: 1, Commitment: 252},
			{Weight: 3, Commitment: 253},
			{Weight: 1, Commitment: 254},
		})
	})

	for _, n := range []int{16, 64, 1024, 2048} {
		// 2048 weight-1 parties saturate the ring: W=2048 makes the stride
		// S=N/W=2, the tightest packing logN=12 supports.
		t.Run(strconv.Itoa(n)+" parties", func(t *testing.T) {
			testWSSLE(t, uniformParties(n))
		})
	}
}

func testWSSLE(t *testing.T, parties []Party) {
	var totalWeight uint64
	for _, p := range parties {
		totalWeight += p.Weight
	}

	params := SetupParams(totalWeight)
	sk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, sk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)

	regs, leaves := registerAll(t, enc, dec, eval, params, parties)

	agg := Aggregate(eval, Identity(enc.Encryptor, params), regs)
	ctOut := Elect(eval, agg, totalWeight)

	pt := dec.DecryptNew(ctOut) // stand-in for ThFHE.Dec(ct_out, I)
	result := DecodeResult(params, pt)

	want := refAggregate(params.RLWE.N(), params.Stride, leaves)

	// h* can come back negated by the ring's negacyclic wraparound, hence
	// Fig. 1 line 16's h* <- W^-1 |h'| (matched by DecodeResult).
	wantCommitment := uint64(math.Round(math.Abs(want.h[0])))

	decoded := DecodeCoeffs(params.RLWE, pt, params.Delta*params.TotalWt)

	wantVec := make([]float64, params.RLWE.N())
	wantVec[0] = want.h[0]
	assertVec(t, "ctOut", decoded, wantVec)

	got := math.Abs(decoded[0])
	t.Logf("commitment precision = %.1f bits (got %v, want %v)",
		-math.Log2(math.Abs(got-float64(wantCommitment))), got, wantCommitment)

	if result != wantCommitment {
		t.Errorf("elected commitment = %d, want %d (r_total=%d, W=%d)", result, wantCommitment, want.r, totalWeight)
	}
	t.Logf("elected commitment: %d (r_total=%d, W=%d)", result, want.r, totalWeight)
}

// uniformParties builds n weight-1 parties with distinct commitments, so the
// total weight is n and every party owns exactly one slot.
func uniformParties(n int) []Party {
	parties := make([]Party, n)
	for i := range parties {
		parties[i] = Party{Weight: 1, Commitment: uint64(250 + i)}
	}
	return parties
}
