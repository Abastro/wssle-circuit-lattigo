package wsslergsw

import (
	"hash/fnv"
	"math"
	"math/big"
	"math/rand"
	"sort"
)

// maxPartyWeight is the largest stake one party may hold in the experiments:
// weights are drawn from [1, maxPartyWeight], out of a total of
// [ParamSet.TotalWeightFor] -- so a party holds at most maxPartyWeight/32 of
// the average stake, and the election is never dominated by one party.
const maxPartyWeight = 50

// randomWeights draws a uniformly random composition of total into n parts,
// each in [1, maxPartyWeight]: every such composition is equally likely.
//
// Stars and bars would give the uniform composition into positive parts, but
// there is no room to reject the ones that break the cap: the mean part is 32
// against a cap of 50, so a uniform composition has a part above the cap with
// probability 1 - (1 - 50/W)^(n-1) ~ 20% *per party*, and almost every draw is
// rejected once n is more than a few dozen.
//
// So the parts are drawn from the tilted distribution p(v) ~ x^v on
// [1, maxPartyWeight], with x fixed by E[v] = total/n ([tiltedWeights]), and
// the sum is corrected by rejection: draw the first n-1 parts, let the last be
// the remainder, reject it if it falls outside the range, and otherwise accept
// the whole vector with probability p(last)/max_v p(v). A surviving vector had
// probability
//
//	prod_i p(w_i) / max_v p(v) = x^total / (Z^n max_v p(v)),
//
// the same for every feasible vector, since the parts sum to total whatever
// they are: the draw is exactly uniform, and the tilt only makes the remainder
// land in range often enough -- an untilted draw would miss total by
// n(maxPartyWeight+1)/2 - total, which is many standard deviations.
func randomWeights(rng *rand.Rand, n int, total uint64) []uint64 {
	if uint64(n) > total || total > uint64(n)*maxPartyWeight {
		panic("no weight assignment of the total into n parts of [1, maxPartyWeight]")
	}

	p := tiltedWeights(float64(total) / float64(n))
	cdf := make([]float64, len(p))
	pMax, sum := 0.0, 0.0
	for v, pv := range p {
		sum += pv
		cdf[v] = sum
		pMax = math.Max(pMax, pv)
	}

	draw := func() uint64 {
		return uint64(sort.SearchFloat64s(cdf, rng.Float64()*cdf[len(cdf)-1]) + 1)
	}

	w := make([]uint64, n)
	for {
		rest := int64(total)
		for i := 0; i < n-1; i++ {
			w[i] = draw()
			rest -= int64(w[i])
		}
		if rest < 1 || rest > maxPartyWeight {
			continue
		}
		if rng.Float64()*pMax > p[rest-1] {
			continue
		}
		w[n-1] = uint64(rest)
		return w
	}
}

// tiltedWeights returns the distribution on [1, maxPartyWeight] proportional
// to x^v whose mean is the one asked for, as p[v-1]. The mean grows with x, so
// the tilt log(x) is found by bisection; the range is wide enough that its ends
// are the point masses at 1 and at maxPartyWeight, which is what a mean at
// either end of the range calls for.
func tiltedWeights(mean float64) []float64 {
	lo, hi := -40.0, 40.0
	for range 200 {
		t := (lo + hi) / 2
		if meanOfTilt(t) < mean {
			lo = t
		} else {
			hi = t
		}
	}
	return tiltProbs((lo + hi) / 2)
}

// tiltProbs is the normalized p(v) ~ exp(t*v) on [1, maxPartyWeight], computed
// against the largest exponent so that a tilt at either end of the bisection
// range underflows to a point mass rather than to NaN.
func tiltProbs(t float64) []float64 {
	p := make([]float64, maxPartyWeight)
	top := math.Max(t, t*maxPartyWeight)
	sum := 0.0
	for v := 1; v <= maxPartyWeight; v++ {
		p[v-1] = math.Exp(t*float64(v) - top)
		sum += p[v-1]
	}
	for v := range p {
		p[v] /= sum
	}
	return p
}

// meanOfTilt is the mean of [tiltProbs].
func meanOfTilt(t float64) float64 {
	mean := 0.0
	for v, pv := range tiltProbs(t) {
		mean += float64(v+1) * pv
	}
	return mean
}

// randomParties builds n parties with distinct commitments at the top of the
// bits-wide range, sharing the total weight as [randomWeights] splits it.
func randomParties(rng *rand.Rand, n int, bits uint, total uint64) []Party {
	top := topCommitment(bits)
	weights := randomWeights(rng, n, total)
	parties := make([]Party, n)
	for i := range parties {
		parties[i] = Party{
			Weight:     weights[i],
			Commitment: new(big.Int).Sub(top, big.NewInt(int64(i))),
		}
	}
	return parties
}

// partyRNG seeds a generator from a case name, so that a given (set, n, trial)
// always draws the same weights -- in the sweep and in the benchmark alike.
func partyRNG(key string) *rand.Rand {
	h := fnv.New64a()
	h.Write([]byte(key))
	return rand.New(rand.NewSource(int64(h.Sum64())))
}
