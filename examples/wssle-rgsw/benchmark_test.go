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
// sizes (16 .. 2048 parties) for each parameter set, mirroring the HIENAA
// reference benchmark so the phases can be compared.
//
// The total weight is the parameter set's W, not n: each of the n parties
// holds W/n, so the packing, the trace depth and the modulus are those of the
// set, and only the party count varies.
//
// Register is per-party work that each party runs on its own machine, so from
// any one party's point of view the latency is a single Register, not n of
// them: the other n-1 registrations are prepared once outside the timer, as are
// the weight ciphertexts, which are public parameters reused across elections.
// The aggregator-side pipeline (Aggregate over all n registrations, Elect,
// Decrypt) is work a party waits on but does not perform, and is reported as
// separate per-phase metrics. Key generation runs once up front, standing in
// for a ThFHE.Setup that a real deployment reuses across elections.
// Single-thread pinned so the breakdown reflects sequential cost.
//
// Commitments are the 32-bit values of [uniformParties] for every set; the
// commitment value does not affect the cost of any phase.
func BenchmarkWSSLE(b *testing.B) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for _, ps := range ParamSets {
		b.Run(ps.Name, func(b *testing.B) {
			params := ps.Params()
			for _, n := range []int{16, 64, 256, 1024, 2048} {
				b.Run(strconv.Itoa(n)+"_parties", func(b *testing.B) {
					benchmarkWSSLE(b, params, n)
				})
			}
		})
	}
}

func benchmarkWSSLE(b *testing.B, params CircuitParams, n int) {
	parties := uniformParties(n)
	for i := range parties {
		parties[i].Weight = params.TotalWt / uint64(n)
	}
	totalWeight := params.TotalWt

	sk, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)
	seed := Identity(enc.Encryptor, params)

	// Stake is a public parameter, encrypted once per weight update rather
	// than per election, so it sits outside the measured path.
	weights := EncryptWeights(enc, params, parties)

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
		agg := Aggregate(eval, seed, weights, regs)
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
