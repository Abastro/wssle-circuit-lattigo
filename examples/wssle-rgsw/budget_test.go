package wsslergsw

import (
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// betaForFailure returns the beta with n*erfc(beta/sqrt2) = 2^-failureBits,
// the tail parameter of prop:err-dec in references/error-analysis. The union
// bound runs over the C coefficients the committee decrypts.
func betaForFailure(n int) float64 {
	target := math.Exp2(-failureBits)
	lo, hi := 1.0, 30.0
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if float64(n)*math.Erfc(mid/math.Sqrt2) > target {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// TestFloodingBudget checks each parameter set's threshold decryption against
// the noise it has to carry:
//
//   - the stored error bound B = 2^LogErrorBound covers beta*sigma_0, with
//     sigma_0 derived afresh from the measured primitives, so the evaluation
//     error exceeds B only with probability 2^-64;
//   - the committee's flooding and B fit below S/2 (FloodingFits, exact);
//   - no fragment wraps Q (MaxFragment, exact).
//
// The flooding bound is hard, so a set passing all three fails decryption only
// in the 2^-64 event that the evaluation error exceeds B.
func TestFloodingBudget(t *testing.T) {
	for _, ps := range ParamSets {
		t.Run(ps.Name, func(t *testing.T) { checkBudget(t, ps) })
	}
}

func checkBudget(t *testing.T, ps ParamSet) {
	params := ps.Params()
	sigma0 := measureSigmaZero(t, params)

	beta := betaForFailure(params.Fragments)
	derived := math.Log2(beta * sigma0)
	t.Logf("sigma_0 = 2^%.2f  beta = %.3f (2^-%d over C = %d)  beta*sigma_0 = 2^%.2f  stored B = 2^%d (%+.2f bits)",
		math.Log2(sigma0), beta, failureBits, params.Fragments, derived, ps.LogErrorBound, float64(ps.LogErrorBound)-derived)
	if derived > float64(ps.LogErrorBound) {
		t.Errorf("set %s: beta*sigma_0 = 2^%.2f exceeds the stored bound 2^%d", ps.Name, derived, ps.LogErrorBound)
	}

	// Flooding: m uniforms on [-F, F] plus B, against S/2.
	m := params.CommitteeSize
	lhs := new(big.Int).Mul(params.FloodBound(), big.NewInt(int64(m)))
	lhs.Add(lhs, new(big.Int).Lsh(big.NewInt(1), uint(ps.LogErrorBound)))
	half := new(big.Int).Rsh(params.ResultScale(), 1)
	t.Logf("m = %d, s = %d, F = 2^%d: m*F + B = 2^%.2f against S/2 = 2^%.0f, headroom %+.2f bits",
		m, params.SmudgeBits(), ps.LogErrorBound+params.SmudgeBits(), bitsOf(lhs), bitsOf(half), bitsOf(half)-bitsOf(lhs))
	if !params.FloodingFits() {
		t.Errorf("set %s: the committee's flooding does not fit below S/2", ps.Name)
	}

	// Ceiling: no stored fragment, h_k + 1 <= 2^H, may wrap Q.
	top := new(big.Int).Lsh(big.NewInt(1), ps.FragmentBits)
	if top.Cmp(params.MaxFragment()) > 0 {
		t.Errorf("set %s: %d-bit fragments wrap Q; MaxFragment has only %d bits",
			ps.Name, ps.FragmentBits, params.MaxFragment().BitLen())
	} else {
		t.Logf("ceiling ok: MaxFragment - 2^%d = 2^%.2f",
			ps.FragmentBits, bitsOf(new(big.Int).Sub(params.MaxFragment(), top)))
	}
}

// measureSigmaZero measures the three primitive variances and derives sigma_0,
// the noise at an output fragment, through the chain of references/error-analysis
// and the pre-multiplied relative trace of [Elect].
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
		eval.ExternalProduct(mono, weight, mono)

		// The encoder as [Aggregate] derives it, RGSW(2*(Y - 1)/(Y - 1)) = 2.
		ringQP := params.RLWE.RingQP().AtLevel(weight.LevelQ(), weight.LevelP())
		ctEcd := rgsw.NewCiphertext(params.RLWE, weight.LevelQ(), weight.LevelP(), 0)
		deriveEncoder(params, thetaQP(params, ringQP), weight, ctEcd)

		deriv := ct.CopyNew()
		eval.ExternalProduct(deriv, ctEcd, deriv)

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

	// The relative trace Tr_{R/Z[Z]}, pre-multiplied by (N/C)^-1: the
	// aggregate's error passes through unamplified, while the key-switching
	// error of doubling step t is doubled by each of the log2(N/C)-1-t steps
	// after it, which sum to ((N/C)^2 - 1)/3 sigma_ks^2 at every fragment
	// alike. sigma_ks^2 is taken as sigma_ext^2: the Galois keys use the same
	// gadget, under sk.
	g := float64(params.RLWE.N() / params.Fragments)
	ks := (g*g - 1) / 3 * sExt
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
