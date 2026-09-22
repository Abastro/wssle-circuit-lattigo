package wsslergsw

import (
	"crypto/rand"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// Threshold decryption (ThFHE.PartDec / ThFHE.FinDec) by the key committee K,
// separate from the parties: m = [CircuitParams.CommitteeSize] members holding
// additive shares of the secret key, all m of whom must take part.
//
// The access structure is m-out-of-m on purpose. A t-out-of-m Shamir sharing
// over varying decryptor sets falls to key recovery however large the flooding
// (Colin de Verdiere, Passelegue, Stehle, eprint 2026/031); with additive
// shares there is one decryptor set and no Lagrange coefficient to vary.
// Secrecy needs one honest member; a withholding member can abort, which the
// protocol answers by accountable abort and a fresh committee -- never by
// re-decrypting the same ciphertext, which would give an honest member's
// second share on the same c1.
//
// Only the C output coefficients are decrypted. Everything else in ct_out is
// zero by construction, so a share covers C coefficients, a few hundred bytes
// rather than a ring element, and only those C coefficients are flooded or
// enter the failure union bound.

// KeyShare is one committee member's additive share of the secret key, over Q,
// in the NTT and Montgomery domain like [rlwe.SecretKey].
type KeyShare struct {
	Value ring.Poly
}

// ShareSecretKey is the trusted dealer's last step: having generated the key
// and every evaluation key from it ([SetupKeys]), it splits the one sparse
// ternary key into m additive shares -- the first m-1 uniform over R_Q, the
// last the difference -- hands share j to member j, and erases the key.
//
// A dealer, rather than lattigo's N-out-of-N key generation
// (multiparty.PublicKeyGenProtocol and GaloisKeyGenProtocol), keeps the joint
// key exactly the Hamming weight 256 secret the noise analysis and the lattice
// estimate assume: under a DKG each member samples its own full-strength key
// and the joint key is their sum, 32 times denser, which puts about 2.5 bits
// on every rounding term.
func ShareSecretKey(params CircuitParams, sk *rlwe.SecretKey) []*KeyShare {
	ringQ := params.RLWE.RingQ()

	prng, err := sampling.NewPRNG()
	if err != nil {
		panic(err)
	}
	uniform := ring.NewUniformSampler(prng, ringQ)

	m := params.CommitteeSize
	shares := make([]*KeyShare, m)
	last := sk.Value.Q.CopyNew()
	for j := 0; j < m-1; j++ {
		// A uniform poly is uniform in any domain, NTT and Montgomery included.
		p := ringQ.NewPoly()
		uniform.Read(p)
		ringQ.Sub(*last, p, *last)
		shares[j] = &KeyShare{Value: p}
	}
	shares[m-1] = &KeyShare{Value: *last}
	return shares
}

// DecryptionShare is one member's partial decryption of the C output
// coefficients, in [0, Q).
type DecryptionShare []*big.Int

// PartialDecrypt is ThFHE.PartDec: coefficients 0 .. C-1 of c1 * sk_j, each
// flooded by the sum of two uniforms on [-F, F], F = [CircuitParams.FloodBound].
//
// Two uniforms rather than one: their sum masks an error of magnitude at most
// B = F / 2^s to statistical distance 2^(-2s), not 2^(-s) (Dahl et al.,
// WAHC'23, eprint 2023/815). And uniform rather than Gaussian: bounded support
// gives the flooding a hard bound, so it can never fail decryption.
func PartialDecrypt(params CircuitParams, ct *rlwe.Ciphertext, share *KeyShare) DecryptionShare {
	ringQ := params.RLWE.RingQ().AtLevel(ct.Level())

	p := ringQ.NewPoly()
	ringQ.MulCoeffsMontgomery(ct.Value[1], share.Value, p)
	ringQ.INTT(p, p)

	Q := ringQ.Modulus()
	F := params.FloodBound()
	out := extractCoeffs(ringQ, p, params.Fragments)
	for _, v := range out {
		v.Add(v, uniformIn(F))
		v.Add(v, uniformIn(F))
		v.Mod(v, Q)
	}
	return out
}

// CombineShares is ThFHE.FinDec up to the rounding: the phase c0 + sum_j of
// the shares, at coefficients 0 .. C-1, centred in (-Q/2, Q/2]. It carries each
// fragment at [CircuitParams.ResultScale], plus the evaluation error and the
// flooding; [DecodeFragments] rounds it off into h*.
func CombineShares(params CircuitParams, ct *rlwe.Ciphertext, shares []DecryptionShare) []*big.Int {
	if len(shares) != params.CommitteeSize {
		panic("m-out-of-m decryption needs every committee member's share")
	}
	ringQ := params.RLWE.RingQ().AtLevel(ct.Level())

	c0 := ringQ.NewPoly()
	ringQ.INTT(ct.Value[0], c0)

	Q := ringQ.Modulus()
	half := new(big.Int).Rsh(Q, 1)
	out := extractCoeffs(ringQ, c0, params.Fragments)
	for k, v := range out {
		for _, s := range shares {
			v.Add(v, s[k])
		}
		v.Mod(v, Q)
		if v.Cmp(half) > 0 {
			v.Sub(v, Q)
		}
	}
	return out
}

// extractCoeffs lifts the first count coefficients of p, in the coefficient
// domain, from RNS to integers in [0, Q) by CRT, Q the modulus at ringQ's level.
func extractCoeffs(ringQ *ring.Ring, p ring.Poly, count int) []*big.Int {
	Q := ringQ.Modulus()
	// AtLevel keeps every prime in SubRings; only the first Level()+1 are live.
	primes := ringQ.SubRings[:ringQ.Level()+1]

	// basis_i = (Q/q_i) * ((Q/q_i)^-1 mod q_i), so x = sum_i r_i basis_i mod Q.
	basis := make([]*big.Int, len(primes))
	for i, s := range primes {
		qi := new(big.Int).SetUint64(s.Modulus)
		Qi := new(big.Int).Quo(Q, qi)
		inv := new(big.Int).ModInverse(new(big.Int).Mod(Qi, qi), qi)
		basis[i] = inv.Mul(inv, Qi)
	}

	out := make([]*big.Int, count)
	r := new(big.Int)
	for k := range out {
		v := new(big.Int)
		for i := range primes {
			v.Add(v, r.Mul(r.SetUint64(p.Coeffs[i][k]), basis[i]))
		}
		out[k] = v.Mod(v, Q)
	}
	return out
}

// uniformIn samples an integer uniformly from [-F, F].
func uniformIn(F *big.Int) *big.Int {
	width := new(big.Int).Lsh(F, 1)
	width.Add(width, big.NewInt(1))
	u, err := rand.Int(rand.Reader, width)
	if err != nil {
		panic(err)
	}
	return u.Sub(u, F)
}
