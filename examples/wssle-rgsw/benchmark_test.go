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
// The aggregator-side pipeline (Aggregate over all n registrations, Elect) is
// work a party waits on but does not perform, and is reported as separate
// per-phase metrics. Threshold decryption is the key committee's, independent
// of n and of the circuit, and has its own [BenchmarkThresholdDecrypt]. Key
// generation runs once up front, standing in for a ThFHE.Setup that a real
// deployment reuses across elections.
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
	_, pk, evk := SetupKeys(params)

	enc := rgsw.NewEncryptor(params.RLWE, pk)
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

	var registerTime, aggregateTime, electTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		regs[0] = Register(enc, params, parties[0])
		registerTime += time.Since(start)

		start = time.Now()
		agg := Aggregate(eval, seed, weights, regs)
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
		copy(coeffs, SplitCommitment(params, topCommitment(params.CommitmentBits())))
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
				DecodeFragments(params, CombineShares(params, ct, shares))
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
