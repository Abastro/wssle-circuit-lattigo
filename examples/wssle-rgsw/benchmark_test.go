package wsslergsw

import (
	"math/big"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// BenchmarkWSSLE measures one party's latency across election sizes of
// 2, 4, 8, .., 2048 parties for each parameter set, the sweep Relect reports.
//
// The total weight is the parameter set's W, not n: each of the n parties
// holds W/n, so the packing, the trace depth and the modulus are those of the
// set, and only the party count varies.
//
// Register is per-party work that each party runs on its own machine, so from
// any one party's point of view the latency is a single Register, not n of
// them: the other n-1 registrations are prepared once outside the timer, as are
// the weight ciphertexts, which are public parameters reused across elections.
// The aggregator-side pipeline (Aggregate over all n registrations, Elect) is
// work a party waits on but does not perform, and is reported as separate
// per-phase metrics. Threshold decryption is the key committee's, independent
// of n and of the circuit, and has its own [BenchmarkThresholdDecrypt]. Key
// generation runs once up front, standing in for a ThFHE.Setup that a real
// deployment reuses across elections.
// Single-thread pinned so the breakdown reflects sequential cost.
//
// All of that untimed setup is built once per (set, n) and cached: Go calls a
// benchmark function twice, a b.N = 1 calibration run and then the requested
// count, and at n = 2048 the setup costs minutes.
//
// Commitments are the 32-bit values of [uniformParties] for every set; the
// commitment value does not affect the cost of any phase.
func BenchmarkWSSLE(b *testing.B) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for _, ps := range ParamSets {
		b.Run(ps.Name, func(b *testing.B) {
			params := ps.Params()
			for n := 2; n <= 2048; n *= 2 {
				b.Run(strconv.Itoa(n)+"_parties", func(b *testing.B) {
					benchmarkWSSLE(b, ps.Name, params, n)
				})
			}
		})
	}
}

// benchSetup is everything outside the timer for one (set, n).
type benchSetup struct {
	parties []Party
	enc     *rgsw.Encryptor
	eval    *rgsw.Evaluator
	weights []*Weight
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

	parties := uniformParties(n)
	for i := range parties {
		parties[i].Weight = params.TotalWt / uint64(n)
	}
	_, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
	st := &benchSetup{
		parties: parties,
		enc:     enc,
		eval:    rgsw.NewEvaluator(params.RLWE, evk),
		// Stake is a public parameter, encrypted once per weight update rather
		// than per election, so it sits outside the measured path.
		weights: EncryptWeights(enc, params, parties),
		regs:    make([]*Registration, n),
	}
	// The other n-1 parties register on their own machines; only party 0's
	// single Register is on the measured latency path.
	for j := 1; j < n; j++ {
		st.regs[j] = Register(enc, params, parties[j])
	}

	benchCache.key, benchCache.setup = key, st
	return st
}

func benchmarkWSSLE(b *testing.B, name string, params CircuitParams, n int) {
	st := setupFor(name, params, n)
	enc, eval, weights, regs, parties := st.enc, st.eval, st.weights, st.regs, st.parties

	var registerTime, aggregateTime, electTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		regs[0] = Register(enc, params, parties[0])
		registerTime += time.Since(start)

		start = time.Now()
		agg := Aggregate(eval, weights, regs)
		aggregateTime += time.Since(start)

		start = time.Now()
		Elect(eval, params, agg)
		electTime += time.Since(start)
	}

	b.ReportMetric(float64(registerTime.Nanoseconds())/float64(b.N), "ns/register")
	b.ReportMetric(float64(aggregateTime.Nanoseconds())/float64(b.N), "ns/aggregate")
	b.ReportMetric(float64(electTime.Nanoseconds())/float64(b.N), "ns/elect")
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
