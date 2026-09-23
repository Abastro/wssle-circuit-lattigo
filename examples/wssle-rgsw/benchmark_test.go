package wsslergsw

import (
	"math/big"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// BenchmarkWSSLE measures one party's latency across election sizes of
// 16, 32, .., 2048 parties for each parameter set, the sweep Relect reports.
//
// The total weight of an n-party election is W = min(32n, N/C)
// ([ParamSet.TotalWeightFor]), split at random over [1, maxPartyWeight]. The
// ring, the trace depth and the modulus are the set's whatever n is; W enters
// only through the slot spacing, and the cost of a phase does not depend on
// it.
//
// What is measured is the aggregator-side pipeline, Aggregate over all n
// registrations and then Elect: work a party waits on but does not perform,
// reported as separate per-phase metrics. Registration is each party's own work
// on its own machine, one Register per party whatever n is, and has its own
// [BenchmarkRegister]; the n registrations and the weight ciphertexts, which
// are public parameters reused across elections, are prepared outside the
// timer. Threshold decryption is the key committee's, independent
// of n and of the circuit, and has its own [BenchmarkThresholdDecrypt]. Key
// generation runs once up front, standing in for a ThFHE.Setup that a real
// deployment reuses across elections.
// Single-thread pinned so the breakdown reflects sequential cost.
//
// All of that untimed setup is built once per (set, n) and cached: Go calls a
// benchmark function twice, a b.N = 1 calibration run and then the requested
// count, and at n = 2048 the setup costs minutes.
//
// Commitments are benchCommitmentBits wide for every set; the commitment value
// does not affect the cost of any phase.
//
// Before the first measurement the process warms up for benchWarmUp, untimed:
// until khugepaged has collapsed the heap into transparent huge pages, about
// one 10 s scan interval into the process, set A runs some 40% slower
// (benchmarks/16).
func BenchmarkWSSLE(b *testing.B) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	warmUpOnce.Do(warmUp)

	for _, ps := range ParamSets {
		b.Run(ps.Name, func(b *testing.B) {
			for n := 16; n <= 2048; n *= 2 {
				params := ps.ParamsFor(n)
				b.Run(strconv.Itoa(n)+"_parties", func(b *testing.B) {
					benchmarkWSSLE(b, ps.Name, params, n)
				})
			}
		})
	}
}

// benchWarmUp is how long [warmUp] runs: one khugepaged scan interval, 10 s,
// with margin.
const benchWarmUp = 15 * time.Second

// warmUpParties is the election size [warmUp] runs, the smallest benchmarked.
const warmUpParties = 16

// benchCommitmentBits is the commitment width the benchmark registers, the
// same for every set: the cost of a phase does not depend on it.
const benchCommitmentBits = 32

var warmUpOnce sync.Once

// warmUp runs set A's smallest election until benchWarmUp has passed, so the
// heap the measurements use has been collapsed into huge pages. Set A is the
// set with the largest polynomials, and the one the slow start was seen on.
func warmUp() {
	params := ParamSetA.ParamsFor(warmUpParties)
	st := setupFor(ParamSetA.Name, params, warmUpParties)
	for start := time.Now(); time.Since(start) < benchWarmUp; {
		st.regs[0] = Register(st.enc, params, st.parties[0])
		Elect(st.eval, params, Aggregate(st.eval, params, st.weights, st.regs))
	}
}

// benchSetup is everything outside the timer for one (set, n).
type benchSetup struct {
	parties []Party
	enc     *rgsw.Encryptor
	eval    *rgsw.Evaluator
	weights []*rgsw.Ciphertext
	regs    []*Registration
}

// benchCache holds the setup of the (set, n) being measured, and only that one:
// set A's at n = 2048 is tens of gigabytes.
var benchCache struct {
	key   string
	setup *benchSetup
}

func setupFor(name string, params CircuitParams, n int) *benchSetup {
	key := name + "/" + strconv.Itoa(n)
	if benchCache.key == key {
		return benchCache.setup
	}
	benchCache.key, benchCache.setup = "", nil // release the previous one first

	// The same weights the sweep checks this (set, n) with.
	parties := randomParties(partyRNG(key), n, benchCommitmentBits, params.TotalWt)
	_, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	st := &benchSetup{
		parties: parties,
		enc:     enc,
		eval:    rgsw.NewEvaluator(params.RLWE, evk),
		// Stake is a public parameter, encrypted once per weight update rather
		// than per election, so its encryption sits outside the measured path.
		// Deriving the encoder from it does not: that belongs to the encoding
		// of a commitment, and [Aggregate] does it per election.
		weights: EncryptWeights(enc, params, parties),
		regs:    make([]*Registration, n),
	}
	// Registration is each party's own work on its own machine, measured by
	// [BenchmarkRegister]; here it is setup, so all n of them run outside the
	// timer.
	for j := range parties {
		st.regs[j] = Register(enc, params, parties[j])
	}

	benchCache.key, benchCache.setup = key, st
	return st
}

func benchmarkWSSLE(b *testing.B, name string, params CircuitParams, n int) {
	st := setupFor(name, params, n)
	eval, weights, regs := st.eval, st.weights, st.regs

	var aggregateTime, electTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		agg := Aggregate(eval, params, weights, regs)
		aggregateTime += time.Since(start)

		start = time.Now()
		Elect(eval, params, agg)
		electTime += time.Since(start)
	}

	b.ReportMetric(float64(aggregateTime.Nanoseconds())/float64(b.N), "ns/aggregate")
	b.ReportMetric(float64(electTime.Nanoseconds())/float64(b.N), "ns/elect")
}

// BenchmarkRegister measures one party's Register, the only phase a party runs
// itself. Nothing about the election enters it: not the party count, not the
// total weight, not the weight the party holds -- only the ring and the
// commitment width, so it is measured once per parameter set.
//
// It is measured on its own because it is short, tens of milliseconds against
// the aggregator's seconds. [BenchmarkWSSLE] runs one election per measurement,
// so a Register inside it is timed a handful of times on a heap that may still
// be cold (benchmarks/16); Go's own b.N loop runs it hundreds of times per
// measurement instead, after the same warm-up.
func BenchmarkRegister(b *testing.B) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	warmUpOnce.Do(warmUp)

	for _, ps := range ParamSets {
		params := ps.Params()
		_, pk, _ := SetupKeys(params)
		enc := rgsw.NewEncryptor(params.RLWE, pk)
		party := Party{
			Weight:     weightPerParty,
			Commitment: topCommitment(benchCommitmentBits),
		}

		b.Run(ps.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Register(enc, params, party)
			}
		})
	}
}

// BenchmarkThresholdDecrypt measures the key committee's decryption, which is
// independent of the circuit and of the number of parties: PartDec is one
// member's work, run by every member in parallel on its own machine; FinDec
// combines all m shares and rounds off h*, and can be run by anyone.
//
// The ciphertext is an encryption of a full-width commitment at the scale of
// [Elect]'s output, the same shape as ct_out, whose content the cost does not
// depend on.
func BenchmarkThresholdDecrypt(b *testing.B) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for _, ps := range ParamSets {
		params := ps.Params()
		sk, pk, _ := SetupKeys(params)
		keyShares := ShareSecretKey(params, sk)

		coeffs := make([]*big.Int, params.RLWE.N())
		for k, f := range storedFragments(params, topCommitment(params.CommitmentBits())) {
			coeffs[params.FragmentIndex(k)] = f
		}
		enc := rlwe.NewEncryptor(params.RLWE, pk)
		ct, err := enc.EncryptNew(encodeScaled(params, coeffs, params.ResultScale()))
		if err != nil {
			b.Fatal(err)
		}

		shares := make([]DecryptionShare, len(keyShares))
		for j, ks := range keyShares {
			shares[j] = PartialDecrypt(params, ct, ks)
		}

		b.Run(ps.Name+"/PartDec", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				PartialDecrypt(params, ct, keyShares[0])
			}
		})
		b.Run(ps.Name+"/FinDec", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := DecodeFragments(params, FinalDecrypt(params, ct, shares)); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// encodeScaled builds the plaintext scale*coeffs for coefficients that may not
// fit a uint64; nil entries are zero.
func encodeScaled(params CircuitParams, coeffs []*big.Int, scale *big.Int) *rlwe.Plaintext {
	pt := rlwe.NewPlaintext(params.RLWE, params.RLWE.MaxLevelQ())
	ringQ := params.RLWE.RingQ()
	qi, v := new(big.Int), new(big.Int)
	for k, c := range coeffs {
		if c == nil {
			continue
		}
		v.Mul(c, scale)
		for j, s := range ringQ.SubRings {
			pt.Value.Coeffs[j][k] = new(big.Int).Mod(v, qi.SetUint64(s.Modulus)).Uint64()
		}
	}
	ringQ.NTT(pt.Value, pt.Value)
	pt.IsNTT = true
	return pt
}
