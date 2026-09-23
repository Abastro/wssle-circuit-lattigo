// Package wsslergsw implements the RGSW variant of the WSSLE ("Pastel")
// single secret leader election circuit -- Fig. 1, p.18 of
// references/Weighted_Single_Secret_Leader_Election_using_CKKS.pdf -- on top
// of Lattigo's core RLWE/RGSW primitives.
//
// It is a port of the HIENAA reference implementation (examples/wssle-rgsw
// there), kept deliberately close to it so the two can be benchmarked against
// each other: same ring degree, same modulus sizes, same secret distribution,
// same circuit shape, same test scenarios.
//
// Everything is coefficient-encoded at the bare [rlwe] level rather than
// through a scheme package: the circuit is only Add, external product and
// automorphism, so there is no rescaling, no relinearization and no
// scheme-level scaling-factor bookkeeping to carry. Messages are plain
// integers scaled by a fixed [CircuitParams.Delta].
//
// Threshold decryption/DKG (ThFHE.Setup/ThFHE.Dec) and the NIZKs of the
// malicious-security protocol are out of scope here, exactly as in the
// reference implementation: a single secret key stands in for the whole key
// committee.
package wsslergsw

import (
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// Each parameter set is sized by two constraints on the scale S = 2*Delta that
// every fragment of [Elect]'s output carries, one from each side
// (references/error-analysis, sec:dec):
//
//	from below   (|K| 2^s + 1) * beta * sigma_0   <=  S/2
//	from above   2^H * S  +  S/2                  <=  Q/2
//
// The lower bound is the flooding budget of threshold decryption (see
// [PartialDecrypt]). Each of the |K| = 32 committee members adds, to each of
// the C output coefficients it decrypts, a uniform on [-2^s B, 2^s B] with
// s = 64: that hides the evaluation error to statistical
// distance about 2^-64 per coefficient, the usual pairing with 128-bit
// computational security, and, having bounded support, it bounds the flooding
// outright. So the only event that can fail decryption is |e_eval| > B, which
// B >= beta*sigma_0, beta = 9.16 to 9.30, makes a 2^-64 event after a union
// bound over the C decrypted coefficients.
//
// A 2^64-fold flood needs a large Q for its scale, and log(QP) is fixed by the
// lattice estimate, so P is small: every set takes twice the gadget digits it
// would otherwise need (d = 2, 4, 2) to keep the decomposition noise down, and
// the external product is correspondingly slower. P also has at least two
// primes in every set: with a single P prime the external product against the
// derived encoder measured 5 bits of sigma above its closed form (set B at
// 159 + 51, d = 4), for a reason not yet understood; split over two primes at
// the same digit ratio, it is 1.3 bits below it, as elsewhere.
//
// sigma_0 is the noise at an output fragment. [Elect] extracts the winner's
// fragments with one relative trace onto Z[Z], Z = X^(N/C), pre-multiplied by
// (N/C)^-1 mod Q: the trace is an exact map on all of R, (N/C) times the
// Z[Z]-component of every element, noise included, so the pre-multiplication
// cancels its gain on the aggregate's error outright, and what the trace adds
// is its own key switching, which the later doubling steps amplify alike at
// every fragment:
//
//	sigma_0^2 = sigma_ecd^2 + ((N/C)^2 - 1)/3 * sigma_ks^2
//
// It is derived, not measured -- an election is a single sample of it -- from
// the measured primitive variances sigma_rlwe, sigma_ext and sigma_ext_tilde,
// each over a fresh key pair and a fresh RGSW operand per sample, with
// sigma_ks^2 taken as sigma_ext^2 (see TestFloodingBudget).
//
// The upper bound keeps each fragment from wrapping Q. Fragments are stored
// offset by one ([EncodeCommitment]), so they reach 2^H, and it caps S at about
// 2^floor(log2 Q - H - 1), with log2 Q the actual size of the prime product:
// lattigo's primes land a little above or below their nominal sizes, which
// moves the floor by one when log2 Q - H - 1 falls just short of an integer, as
// it does for set C, whose Delta is 2^97 rather than 2^98 for that reason.
// [CircuitParams.MaxFragment] reports the bound exactly.
//
// Lattice security is from the full LWE.estimate of the lattice estimator, not
// the rough one, which runs only usvp and dual_hybrid and so misses
// bdd_mitm_hybrid, the attack that binds for this sparse ternary secret:
//
//	set  log(QP)     security  d  C x H   sigma_0   B      flooding headroom
//	A    236 + 180   128.0     2  1x128   2^15.4    2^19   +18 bits
//	B    159 + 51    127.9     4  2x64    2^14.3    2^18   +6 bits
//	C    132 + 77    128.5     2  4x32    2^13.2    2^17   +11 bits
//	ALow 200 + 216   128.0     1  1x128   2^15.4    2^19   +6 bits   (s = 40)
//	BLow 131 + 79    127.9     2  2x64    2^13.5    2^17   +3 bits   (s = 40)
//	CLow 98 + 111    128.5     1  4x32    2^12.5    2^16   +2 bits   (s = 40)
//
// In set A it is the trace's key switching that sets sigma_0. In sets B and C
// the relative trace's key switching, ((N/C)^2 - 1)/3 sigma_ks^2, is C^2 smaller
// than the full trace's, and the aggregate's own error, raised by the C-times
// longer theta, is as large.
//
// Unlike the CKKS variant of the same circuit, no multiplicative level chain is
// needed: the external product consumes no levels, so the modulus is sized
// purely by the noise budget, not by how many parties fold into [Aggregate].
const (
	hammingWeight = 256
	sigma         = 3.2

	// failureBits bounds the decryption failure probability, 2^-failureBits,
	// through B (see [ParamSet.LogErrorBound]). It is the same for every set,
	// whatever its [ParamSet.StatSecurity].
	failureBits = 64
)

// ParamSet is one row of the paper's parameter table.
type ParamSet struct {
	Name       string
	LogN       int
	LogQ, LogP []int // bit sizes of the Q and P primes; d = ceil(len(LogQ)/len(LogP))
	LogDelta   int
	// A commitment is split into Fragments (the paper's C) fragments of
	// FragmentBits (its H) bits each. With S = N/(C*W), Y = X^S and
	// Z = Y^W = X^(N/C), slot j is Y^j and carries its k-th fragment at
	// Y^(j + k*W) = X^(S*(j + k*W)): the slots S apart, the fragments of a slot
	// N/C apart, on the subring Z[Z] (references/circuit). C must be a power
	// of two, and C*W must divide N.
	Fragments    int
	FragmentBits uint

	// CommitteeSize is m, the size of the key committee, which is separate from
	// the parties and decrypts m-out-of-m.
	CommitteeSize int
	// StatSecurity is s, the statistical security of threshold decryption: the
	// flooding hides the evaluation error at each decrypted coefficient to
	// statistical distance about 2^-s ([CircuitParams.SmudgeBits]).
	StatSecurity int
	// LogErrorBound is log2 B, a bound on the evaluation error at an output
	// fragment holding except with probability 2^-64: B >= beta*sigma_0 with
	// C*erfc(beta/sqrt2) = 2^-64, sigma_0 as derived by TestFloodingBudget,
	// rounded up to a power of two. It sizes the flooding.
	LogErrorBound int
}

// The parameter sets of the paper, all for 128-bit commitments and a committee
// of 32: A, B and C at 64-bit statistical security.
var (
	ParamSetA = ParamSet{
		Name: "A", LogN: 14,
		LogQ: []int{40, 40, 39, 39, 39, 39}, LogP: []int{60, 60, 60}, // 236 + 180, d = 2
		LogDelta: 106, Fragments: 1, FragmentBits: 128,
		CommitteeSize: 32, StatSecurity: 64, LogErrorBound: 19,
	}
	ParamSetB = ParamSet{
		Name: "B", LogN: 13,
		LogQ: []int{20, 20, 20, 20, 20, 20, 20, 19}, LogP: []int{26, 25}, // 159 + 51, d = 4
		LogDelta: 93, Fragments: 2, FragmentBits: 64,
		CommitteeSize: 32, StatSecurity: 64, LogErrorBound: 18,
	}
	ParamSetC = ParamSet{
		Name: "C", LogN: 13,
		LogQ: []int{33, 33, 33, 33}, LogP: []int{39, 38}, // 132 + 77, d = 2
		LogDelta: 97, Fragments: 4, FragmentBits: 32,
		CommitteeSize: 32, StatSecurity: 64, LogErrorBound: 17,
	}

	// The same three at 40-bit statistical security, s = 40, on the layouts the
	// smaller flood allows: fewer gadget digits (d = 1, 2, 1), so a faster
	// external product. Same ring, log(QP) and so lattice security as A, B, C
	// (416, 210, 209), and the same 2^-64 failure probability.
	ParamSetALow = ParamSet{
		Name: "ALow", LogN: 14,
		LogQ: []int{50, 50, 50, 50}, LogP: []int{54, 54, 54, 54}, // 200 + 216, d = 1
		LogDelta: 70, Fragments: 1, FragmentBits: 128,
		CommitteeSize: 32, StatSecurity: 40, LogErrorBound: 19,
	}
	ParamSetBLow = ParamSet{
		Name: "BLow", LogN: 13,
		LogQ: []int{33, 33, 33, 32}, LogP: []int{40, 39}, // 131 + 79, d = 2
		LogDelta: 65, Fragments: 2, FragmentBits: 64,
		CommitteeSize: 32, StatSecurity: 40, LogErrorBound: 17,
	}
	ParamSetCLow = ParamSet{
		Name: "CLow", LogN: 13,
		LogQ: []int{49, 49}, LogP: []int{56, 55}, // 98 + 111, d = 1
		LogDelta: 63, Fragments: 4, FragmentBits: 32,
		CommitteeSize: 32, StatSecurity: 40, LogErrorBound: 16,
	}

	ParamSets = []ParamSet{ParamSetA, ParamSetB, ParamSetC, ParamSetALow, ParamSetBLow, ParamSetCLow}
)

// MaxTotalWeight is the largest public total weight the set's layout admits,
// W = N/C: the slots are S = N/(C*W) apart, so C*W has to divide N, and the
// fragments of a slot sit N/C apart whatever W is.
func (ps ParamSet) MaxTotalWeight() uint64 {
	return uint64((1 << ps.LogN) / ps.Fragments)
}

// Params builds the [CircuitParams] of the set at its largest total weight.
// The noise grows with W (theta carries C*W), so this is the set as it was
// sized, and [ParamSet.ParamsFor] at any election size is inside it.
func (ps ParamSet) Params() CircuitParams {
	return ps.paramsAt(ps.MaxTotalWeight())
}

// paramsAt builds the [CircuitParams] of the set at a given total weight.
func (ps ParamSet) paramsAt(W uint64) CircuitParams {
	N := 1 << ps.LogN
	C := ps.Fragments
	if C < 1 || C&(C-1) != 0 {
		panic("fragments per commitment must be a power of two")
	}
	if W == 0 || N%(C*int(W)) != 0 {
		panic("fragments times total weight must divide the ring degree")
	}

	params, err := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN:    ps.LogN,
		LogQ:    ps.LogQ,
		LogP:    ps.LogP,
		Xs:      ring.Ternary{H: hammingWeight},
		Xe:      ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma},
		NTTFlag: true,
	})
	if err != nil {
		panic(err)
	}

	cp := CircuitParams{
		RLWE:          params,
		Delta:         new(big.Int).Lsh(big.NewInt(1), uint(ps.LogDelta)),
		StatSecurity:  ps.StatSecurity,
		Stride:        N / (C * int(W)),
		TotalWt:       W,
		Fragments:     ps.Fragments,
		FragmentBits:  ps.FragmentBits,
		CommitteeSize: ps.CommitteeSize,
		LogErrorBound: ps.LogErrorBound,
	}
	if ps.CommitteeSize < 1 {
		panic("the key committee needs at least one member")
	}
	if !cp.FloodingFits() {
		panic("the flooding of a full committee does not fit the output scale")
	}
	return cp
}

// weightPerParty is the average stake an experiment gives a party: an n-party
// election runs at W = 32n, until that reaches the layout's ceiling. Party
// weights are then drawn at random from [1, 50], so no party holds much more
// than its share (see randomWeights).
const weightPerParty = 32

// TotalWeightFor is the public total weight of an n-party election,
// W = min(32n, N/C), the ceiling being [ParamSet.MaxTotalWeight]. Both ends
// are powers of two only if n is, which every election size the experiments
// use is.
func (ps ParamSet) TotalWeightFor(n int) uint64 {
	if w := uint64(weightPerParty * n); w < ps.MaxTotalWeight() {
		return w
	}
	return ps.MaxTotalWeight()
}

// ParamsFor is [ParamSet.Params] at the total weight of an n-party election.
// A smaller W only ever helps the noise -- theta carries C*W -- so a set sized
// at its ceiling holds at every n.
func (ps ParamSet) ParamsFor(n int) CircuitParams {
	return ps.paramsAt(ps.TotalWeightFor(n))
}

// SetupParams builds [CircuitParams] on set C's ring, modulus and fragment
// layout for an election of any other public total weight, which must be a
// power of two with N/W at least set C's four fragments. The step tests use it
// with small W.
func SetupParams(totalWeight uint64) CircuitParams {
	return ParamSetC.paramsAt(totalWeight)
}

// CircuitParams holds the ring parameters and packing layout for the circuit.
type CircuitParams struct {
	RLWE rlwe.Parameters
	// Delta is the scaling factor applied to commitment fragments. It is not a
	// free knob: [CircuitParams.ResultScale] has to leave room for the flooding
	// noise below and [CircuitParams.MaxFragment] above (see [ParamSet]).
	Delta   *big.Int
	Stride  int    // S = N/(C*W), the slot spacing: slot j is Y^j, Y = X^S.
	TotalWt uint64 // W, the public total weight.

	Fragments    int  // C, the fragments per commitment
	FragmentBits uint // H, the bits per fragment

	CommitteeSize int // m, the key committee
	StatSecurity  int // s, see [ParamSet.StatSecurity]
	LogErrorBound int // log2 B, see [ParamSet.LogErrorBound]
}

// FragmentIndex is the coefficient of fragment k of slot 0: Z^k = X^(k*N/C).
// Slot j carries its fragment k at Y^j times that. It is also where [Elect]
// leaves the winner's fragments, rotated by an unknown Z^c ([DecodeFragments]).
func (p CircuitParams) FragmentIndex(k int) int {
	return k * (p.RLWE.N() / p.Fragments)
}

// CommitmentBits is the width of a commitment, C*H.
func (p CircuitParams) CommitmentBits() uint {
	return uint(p.Fragments) * p.FragmentBits
}

// Scale is the factor a freshly encrypted fragment carries: Delta alone.
func (p CircuitParams) Scale() *big.Int {
	return new(big.Int).Set(p.Delta)
}

// EncodedScale is the factor a party's contribution carries once [encodeH] has
// applied the 2/(Y-1) of [deriveEncoder]: 2*Delta.
func (p CircuitParams) EncodedScale() *big.Int {
	return new(big.Int).Lsh(p.Delta, 1)
}

// ResultScale is the factor each fragment of [Elect]'s output carries: the
// same 2*Delta as the aggregate. The trace would multiply it by N, but [Elect]
// pre-multiplies by N^-1 mod Q, which cancels that exactly.
func (p CircuitParams) ResultScale() *big.Int {
	return p.EncodedScale()
}

// SmudgeBits is s, the statistical distance 2^-s the flooding leaves at each
// decrypted coefficient, the set's [ParamSet.StatSecurity].
func (p CircuitParams) SmudgeBits() int {
	return p.StatSecurity
}

// FloodSigma is sigma_flood = 2^(s-1) * B, the standard deviation each member
// floods each coefficient of its share with ([gaussianFlood]).
//
// Shifting a Gaussian of standard deviation sigma by at most B leaves
// statistical distance about B/(sigma*sqrt(2pi)), so this sigma hides an
// evaluation error of magnitude B to 2/sqrt(2pi) * 2^-s = 0.80 * 2^-s per
// coefficient, inside the 2^-s asked for. Gaussian rather than uniform because
// the m members' floods then add in variance: the committee costs
// sqrt(m)*sigma rather than m*F, half a bit per doubling of m instead of one.
func (p CircuitParams) FloodSigma() *big.Int {
	return new(big.Int).Lsh(big.NewInt(1), uint(p.LogErrorBound+p.SmudgeBits()-1))
}

// FloodTailFactor is ceil(beta*sqrt(m)), the multiple of sigma_flood the whole
// committee's flooding stays within except with probability 2^-failureBits:
// the m floods sum to standard deviation sqrt(m)*sigma_flood, and beta is the
// tail of [betaForFailure] over the C decrypted coefficients.
func (p CircuitParams) FloodTailFactor() int64 {
	return int64(math.Ceil(betaForFailure(p.Fragments) * math.Sqrt(float64(p.CommitteeSize))))
}

// FloodingFits reports whether the whole committee's flooding, together with
// the evaluation error, stays below the S/2 that rounding tolerates:
//
//	ceil(beta*sqrt(m)) * sigma_flood + B <= S/2.
//
// Neither side of the failure budget is spent twice: the evaluation error
// exceeds B with probability 2^-failureBits, and the flooding exceeds its own
// term with the same probability, so a set passing this check fails decryption
// with probability at most 2^(1-failureBits). In practice the flooding term is
// far from binding -- every set has bits of headroom, and [gaussianFlood]'s
// support is bounded by 6*sigma_flood per member anyway -- so the failure
// probability is the evaluation error's alone.
func (p CircuitParams) FloodingFits() bool {
	lhs := new(big.Int).Mul(p.FloodSigma(), big.NewInt(p.FloodTailFactor()))
	lhs.Add(lhs, new(big.Int).Lsh(big.NewInt(1), uint(p.LogErrorBound)))
	return lhs.Cmp(new(big.Int).Rsh(p.ResultScale(), 1)) <= 0
}

// betaForFailure returns the beta with n*erfc(beta/sqrt2) = 2^-failureBits,
// the tail parameter of prop:err-dec in references/error-analysis. The union
// bound runs over the C coefficients the committee decrypts.
func betaForFailure(n int) float64 {
	target := math.Exp2(-failureBits)
	lo, hi := 1.0, 30.0
	for range 200 {
		mid := (lo + hi) / 2
		if float64(n)*math.Erfc(mid/math.Sqrt2) > target {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// MaxFragment is the largest stored fragment, h_k + 1 ([EncodeCommitment]),
// that [Elect]'s output can carry without wrapping Q: the largest h with
// h*S + S/2 <= Q/2, S = [CircuitParams.ResultScale], where S/2 is the most
// error the final rounding tolerates. The same bound covers the aggregate,
// whose coefficients carry stored fragments at the same scale.
func (p CircuitParams) MaxFragment() *big.Int {
	scale := p.ResultScale()
	h := new(big.Int).Rsh(p.RLWE.RingQ().Modulus(), 1) // floor(Q/2): Q is odd
	h.Sub(h, new(big.Int).Rsh(scale, 1))
	return h.Div(h, scale)
}

// SetupKeys generates the secret key, the public key parties encrypt under,
// and the automorphism keys of the full trace [Elect] applies.
//
// There is no relinearization key: this variant never multiplies two
// ciphertexts anywhere (see [Elect]), so Galois keys are the only evaluation
// keys in play.
//
// This is the trusted dealer: it generates every key from one secret key, and
// then hands the committee additive shares of that key ([ShareSecretKey]) and
// erases it. The circuit only ever sees pk and the evaluation keys; decryption
// goes through the shares.
//
// Registration must encrypt under pk, not sk: a party does not hold the secret
// key. The distinction is not cosmetic for the noise. A secret-key RGSW
// ciphertext carries error variance sigma_err^2, whereas a public-key one
// carries sigma_err^2*(1 + N*sigma_Enc^2 + N*sigma_sk^2) = sigma_err^2*(2h+1),
// a factor of 513 at h=256 -- 4.5 bits of sigma, straight through the circuit
// and onto the modulus (see the sizing note above).
func SetupKeys(params CircuitParams) (*rlwe.SecretKey, *rlwe.PublicKey, rlwe.EvaluationKeySet) {
	kgen := rlwe.NewKeyGenerator(params.RLWE)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	gks := kgen.GenGaloisKeysNew(ExtractionGaloisElements(params), sk)
	return sk, pk, rlwe.NewMemEvaluationKeySet(nil, gks...)
}
