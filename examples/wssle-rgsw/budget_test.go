package wsslergsw

import (
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Statistical parameters of the threshold decryption, settled together:
//
//	statBits  per-uniform smudging exponent. The flooding each committee
//	          member adds is the sum of TWO uniforms on [-2^statBits*B,
//	          2^statBits*B], which by Dahl et al. (WAHC'23, eprint 2023/815)
//	          gives statistical distance 2^(-2*statBits) rather than the
//	          2^(-statBits) a single uniform would give.
//	failBits  target decryption failure probability, 2^-failBits. The tail
//	          parameter beta solving N*erfc(beta/sqrt2) = 2^-failBits is derived
//	          per set by betaForFailure, since the union bound of
//	          prop:err-dec runs over the ring's N coefficients.
//
// Committee members each contribute two uniforms, so the flooding sum has the
// hard bound 2*|K| * 2^statBits * B = 2^(statBits + 1 + log2|K|) * B. Nothing
// about it is probabilistic: uniform smudging has bounded support, so the only
// event that can fail decryption is |e_eval| > B.
const (
	statBits      = 32
	committeeSize = 32
	failBits      = 64
)

// betaForFailure returns the beta with N*erfc(beta/sqrt2) = 2^-failBits, the
// tail parameter of prop:err-dec in references/error-analysis.
func betaForFailure(N int) float64 {
	target := math.Exp2(-failBits)
	lo, hi := 1.0, 30.0
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if float64(N)*math.Erfc(mid/math.Sqrt2) > target {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// TestFloodingBudget derives sigma_0 for each parameter set from the measured
// primitives and checks the one sizing rule that threshold decryption imposes:
//
//	(2 * |K| * 2^statBits + 1) * B  <=  scale/2,   B = beta * sigma_0
//
// The left side is a hard bound, so a set that clears it never fails on the
// flooding; a set that does not clear it always fails.
func TestFloodingBudget(t *testing.T) {
	for _, ps := range ParamSets {
		t.Run(ps.Name, func(t *testing.T) { checkBudget(t, ps) })
	}
}

// checkBudget applies both sides of the scale constraint to one parameter set:
// the flooding budget from below, and the no-wrap ceiling from above.
func checkBudget(t *testing.T, ps ParamSet) {
	floodBits := statBits + 1 + int(math.Log2(committeeSize)) // 2*|K| uniforms

	params := ps.Params()
	sigma0 := measureSigmaZero(t, params)

	beta := betaForFailure(params.RLWE.N())
	B := beta * sigma0
	// need = 2^floodBits * B + B, in bits
	need := math.Log2(math.Exp2(float64(floodBits))*B+B) + 1 // +1: scale/2
	have := float64(params.ResultScale().BitLen() - 1)       // log2(scale)

	t.Logf("sigma_0 = 2^%.2f   beta = %.3f (p_fail = 2^-%d over N = %d)   B = 2^%.2f",
		math.Log2(sigma0), beta, failBits, params.RLWE.N(), math.Log2(B))
	t.Logf("flooding hard bound = 2^%d * B = 2^%.2f", floodBits, float64(floodBits)+math.Log2(B))
	t.Logf("scale needed >= 2^%.2f   have 2^%.0f   headroom %+.2f bits", need, have, have-need)

	// Upper bound: no fragment may wrap Q, which is what
	// [CircuitParams.MaxFragment] bounds, exactly.
	top := topCommitment(ps.FragmentBits)
	if top.Cmp(params.MaxFragment()) > 0 {
		t.Errorf("set %s: %d-bit fragments wrap Q; MaxFragment has only %d bits",
			ps.Name, ps.FragmentBits, params.MaxFragment().BitLen())
	} else {
		t.Logf("ceiling ok: MaxFragment - (2^%d - 1) = 2^%.2f",
			ps.FragmentBits, bitsOf(new(big.Int).Sub(params.MaxFragment(), top)))
	}

	if have < need {
		t.Errorf("set %s: scale 2^%.0f is %.2f bits short of the flooding budget", ps.Name, have, need-have)
	}
}

// measureSigmaZero measures the three primitive variances and derives sigma_0,
// the noise at an output fragment, through the chain of references/error-analysis
// and the pre-multiplied full trace of [Elect].
func measureSigmaZero(t *testing.T, params CircuitParams) float64 {
	const reps = 256

	W := float64(params.TotalWt)
	n := W // uniform weight-1 parties

	kgen := rlwe.NewKeyGenerator(params.RLWE)

	// A fresh key pair per rep. The external product needs no Galois keys, so
	// this skips SetupKeys' trace keys, which dominate key generation. Without
	// it every rep shares one (sk, pk) and the whole run is a single sample of
	// whatever the primitives depend on through the key -- which is what kept
	// the spread at ~0.8 bits no matter how many reps were added.
	sample := func() (sRLWE, sMono, sDeriv float64) {
		sk := kgen.GenSecretKeyNew()
		pk := kgen.GenPublicKeyNew(sk)

		enc := rgsw.NewEncryptor(params.RLWE, pk)
		dec := rlwe.NewDecryptor(params.RLWE, sk)
		eval := rgsw.NewEvaluator(params.RLWE, nil)

		noise := func(ct *rlwe.Ciphertext, scale *big.Int) float64 {
			return rms(Residuals(params.RLWE, dec.DecryptNew(ct), scale))
		}

		weight := EncryptWeight(enc, params, 1)
		ct := Identity(enc.Encryptor, params)

		mono := ct.CopyNew()
		eval.ExternalProduct(mono, weight.CtW, mono)

		deriv := ct.CopyNew()
		eval.ExternalProduct(deriv, weight.CtEcd, deriv)

		return sq(noise(ct, params.Scale())),
			sq(noise(mono, params.Scale())),
			sq(noise(deriv, params.EncodedScale()))
	}

	var vIn, vMono, vDeriv float64
	for i := 0; i < reps; i++ {
		a, b, c := sample()
		vIn, vMono, vDeriv = vIn+a, vMono+b, vDeriv+c
	}
	vIn, vMono, vDeriv = vIn/reps, vMono/reps, vDeriv/reps

	// Only two primitives are measured: sigma_rlwe from a fresh encryption and
	// sigma_ext from one external product against an RGSW monomial, whose
	// message has ||mu'||_2^2 = 1.
	sRLWE, sExt := vIn, vMono-vIn

	// sigma_ext_tilde is measured too, against the derived encoder, whose message
	// 2(1 + Y + ... + Y^(w-1)) has ||mu'||_2^2 = 4w -- here 4, at w = 1.
	sExtTilde := vDeriv - 4*vIn

	t.Logf("  sigma_rlwe = 2^%.2f  sigma_ext = 2^%.2f  sigma_ext_tilde = 2^%.2f  (measured)",
		lg(sRLWE), lg(sExt), lg(sExtTilde))
	t.Logf("  cross-check: closed-form sigma_ext_tilde = 2^%.2f (D = %.4g)",
		lg(sExt+(W-1)*decompositionVariance(params)), decompositionVariance(params))

	// sigma_ecd^2 <= (4W+1) sigma_rlwe^2 + n sigma_ext_tilde^2 + 2n sigma_ext^2
	ecd := (4*W+1)*sRLWE + n*sExtTilde + 2*n*sExt

	// The full trace, pre-multiplied by N^-1: the aggregate's error passes
	// through unamplified, while the key-switching error of doubling step t is
	// doubled by each of the log2(N)-1-t steps after it, which sum to
	// (N^2 - 1)/3 sigma_ks^2 at the constant coefficient. sigma_ks^2 is taken
	// as sigma_ext^2: the Galois keys use the same gadget, with less error.
	N := float64(params.RLWE.N())
	ks := (N*N - 1) / 3 * sExt
	t.Logf("  sigma_ecd = 2^%.2f  trace key switching = 2^%.2f", lg(ecd), lg(ks))
	return math.Sqrt(ecd + ks)
}

// decompositionVariance returns D = (2dNB^2/3p^2) * sigma_rgsw^2, the part of an
// external product's error that comes from the RGSW operand's own noise passing
// through the gadget decomposition. Everything in it is fixed by the parameters:
// d digits, digit base B, auxiliary modulus p = P, and sigma_rgsw^2 =
// sigma_err^2 (2h+1) for a public-key RGSW ciphertext (Definition 2).
func decompositionVariance(params CircuitParams) float64 {
	p := params.RLWE
	levelQ, levelP := p.MaxLevelQ(), p.MaxLevelP()

	// Digits group levelP+1 consecutive Q primes; B is the largest digit modulus.
	perDigit := levelP + 1
	qi := p.Q()
	maxDigit := new(big.Float)
	for i := 0; i <= levelQ; i += perDigit {
		digit := new(big.Int).SetUint64(1)
		for j := i; j < i+perDigit && j <= levelQ; j++ {
			digit.Mul(digit, new(big.Int).SetUint64(qi[j]))
		}
		if f := new(big.Float).SetInt(digit); f.Cmp(maxDigit) > 0 {
			maxDigit = f
		}
	}

	pBig := new(big.Float).SetInt(p.RingP().Modulus())
	ratio, _ := new(big.Float).Quo(maxDigit, pBig).Float64() // B/p

	d := float64(p.BaseRNSDecompositionVectorSize(levelQ, levelP))
	N := float64(p.N())
	sigmaRGSW2 := sigma * sigma * float64(2*hammingWeight+1)

	return 2 * d * N * ratio * ratio / 3 * sigmaRGSW2
}

// bitsOf reports log2 of a big.Int with enough precision to see the boundary,
// which a float64 conversion loses once the value passes 2^53.
func bitsOf(x *big.Int) float64 {
	e := x.BitLen()
	// frac = x / 2^e in [0.5, 1)
	frac, _ := new(big.Float).Quo(
		new(big.Float).SetInt(x),
		new(big.Float).SetMantExp(big.NewFloat(1), e),
	).Float64()
	return float64(e) + math.Log2(frac)
}
