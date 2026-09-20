package wsslergsw

import (
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// BenchmarkWSSLE measures one party's latency across a range of election
// sizes (16 .. 2048 weight-1 parties), mirroring the HIENAA reference
// benchmark so the two libraries can be compared phase by phase.
//
// Register is per-party work that each party runs on its own machine, so from
// any one party's point of view the latency is a single Register, not n of
// them: the other n-1 registrations are prepared once outside the timer.
// The aggregator-side pipeline (Aggregate over all n registrations, Elect,
// Decrypt) is work a party waits on but does not perform, and is reported as
// separate per-phase metrics. SetupParams/SetupKeys run once up front,
// standing in for a ThFHE.Setup that a real deployment reuses across
// elections. Single-thread pinned so the breakdown reflects sequential cost.
func BenchmarkWSSLE(b *testing.B) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for _, n := range []int{16, 64, 256, 1024, 2048} {
		b.Run(strconv.Itoa(n)+" parties", func(b *testing.B) {
			benchmarkWSSLE(b, n)
		})
	}
}

func benchmarkWSSLE(b *testing.B, n int) {
	parties := uniformParties(n)

	var totalWeight uint64
	for _, p := range parties {
		totalWeight += p.Weight
	}

	params := SetupParams(totalWeight)
	sk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, sk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)
	seed := Identity(enc.Encryptor, params)

	// The other n-1 parties register on their own machines; prepare their
	// registrations once, outside the timer, so only party 0's single
	// Register is on the measured latency path.
	regs := make([]*Registration, len(parties))
	for j := 1; j < len(parties); j++ {
		regs[j] = Register(enc, params, parties[j])
	}

	var registerTime, aggregateTime, electTime, decryptTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		regs[0] = Register(enc, params, parties[0])
		registerTime += time.Since(start)

		start = time.Now()
		agg := Aggregate(eval, seed, regs)
		aggregateTime += time.Since(start)

		start = time.Now()
		ctOut := Elect(eval, agg, totalWeight)
		electTime += time.Since(start)

		start = time.Now()
		dec.DecryptNew(ctOut)
		decryptTime += time.Since(start)
	}

	b.ReportMetric(float64(registerTime.Nanoseconds())/float64(b.N), "ns/register")
	b.ReportMetric(float64(aggregateTime.Nanoseconds())/float64(b.N), "ns/aggregate")
	b.ReportMetric(float64(electTime.Nanoseconds())/float64(b.N), "ns/elect")
	b.ReportMetric(float64(decryptTime.Nanoseconds())/float64(b.N), "ns/decrypt")
}
