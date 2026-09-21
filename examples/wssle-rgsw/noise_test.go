package wsslergsw

import (
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// TestNoiseModel measures the three primitive error variances and feeds them
// through the chain of references/error-analysis to predict sigma_0, the noise
// at the constant coefficient of [Elect]'s output.
//
// The primitives are what should be measured: one ciphertext carries N
// coefficients, so a handful of encryptions gives thousands of samples. sigma_0
// is what should be derived: a whole election yields exactly one sample of it,
// and the trace makes it sqrt(W) larger than the bulk it would be read off.
//
// The measured sigma_rlwe is smaller than Definition 2 of the paper predicts,
// because lattigo encrypts over QP and rescales by P, leaving only the rounding
// (1+N*sigma_sk^2)/12 rather than sigma_err^2*(2h+1). That is a legitimate
// implementation choice which the analysis does not have to assume.
func TestNoiseModel(t *testing.T) {
	const reps = 8

	params := SetupParams(2048)
	W := float64(params.TotalWt)
	n := W // uniform weight-1 parties

	sk, pk, evk := SetupKeys(params)
	enc := rgsw.NewEncryptor(params.RLWE, pk)
	dec := rlwe.NewDecryptor(params.RLWE, sk)
	eval := rgsw.NewEvaluator(params.RLWE, evk)

	noise := func(ct *rlwe.Ciphertext, scale *big.Int) float64 {
		return rms(Residuals(params.RLWE, dec.DecryptNew(ct), scale))
	}

	// One monomial RGSW and one derived encoder, both for weight 1.
	weight := EncryptWeight(enc, params, 1)

	var vIn, vMono, vDeriv float64
	for i := 0; i < reps; i++ {
		ct := Identity(enc.Encryptor, params) // fresh Enc(0) under pk
		vIn += sq(noise(ct, params.Scale()))

		mono := ct.CopyNew()
		eval.ExternalProduct(mono, weight.CtW, mono)
		vMono += sq(noise(mono, params.Scale()))

		deriv := ct.CopyNew()
		eval.ExternalProduct(deriv, weight.CtEcd, deriv)
		vDeriv += sq(noise(deriv, params.EncodedScale()))
	}
	vIn, vMono, vDeriv = vIn/reps, vMono/reps, vDeriv/reps

	// sigma_out^2 = ||mu'||_2^2 * sigma_in^2 + sigma_ext^2, with ||mu'||^2 = 1
	// for the monomial Y^w and 4 for the derived encoder's 2*(1+...+Y^(w-1)).
	sRLWE, sExt, sExtTilde := vIn, vMono-vIn, vDeriv-4*vIn

	t.Logf("measured over %d reps x %d coefficients:", reps, params.RLWE.N())
	t.Logf("  sigma_rlwe      = 2^%.2f   (Definition 2 would give 2^%.2f)",
		lg(sRLWE), lg(3.2*3.2*(2*hammingWeight+1)))
	t.Logf("  sigma_ext       = 2^%.2f", lg(sExt))
	t.Logf("  sigma_ext_tilde = 2^%.2f   (+%.2f bits from ||theta||_2^2 = W)",
		lg(sExtTilde), lg(sExtTilde)-lg(sExt))

	// sigma_ecd^2 <= (4W+1) sigma_rlwe^2 + n sigma_ext_tilde^2 + 2n sigma_ext^2
	// sigma_0     <= W sqrt(sigma_ecd^2 + sigma_ext^2/3)
	ecd := (4*W+1)*sRLWE + n*sExtTilde + 2*n*sExt
	t.Logf("  => sigma_0      = 2^%.2f  (derived, n = W = %.0f)", lg(W*W*(ecd+sExt/3)), W)
}

func sq(x float64) float64 { return x * x }

// lg reports a variance in bits of standard deviation.
func lg(variance float64) float64 { return 0.5 * math.Log2(variance) }

// rms is the root-mean-square of a residual vector.
func rms(res []*big.Int) float64 {
	sum := new(big.Float)
	for _, v := range res {
		f := new(big.Float).SetInt(v)
		sum.Add(sum, f.Mul(f, f))
	}
	ms, _ := sum.Quo(sum, big.NewFloat(float64(len(res)))).Float64()
	return math.Sqrt(ms)
}
