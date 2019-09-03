package tdigest

import (
	"math"
	"sort"
)

func cdf(x float64, data []float64) float64 {
	sorted := getSortedCopy(data)
	return cdfOnSorted(x, sorted)
}

func cdfOnSorted(x float64, data []float64) float64 {
	n1 := float64(0)
	n2 := float64(0)
	for _, v := range data {
		if v < x {
			n1++
		}
		if v == x {
			n2++
		}
	}
	return (n1 + n2/2.0) / float64(len(data))
}

func quantile(q float64, data []float64) float64 {
	sorted := getSortedCopy(data)
	return quantileOnSorted(q, sorted)
}

func quantileOnSorted(q float64, data []float64) float64 {
	n := float64(len(data))
	if n == 0 {
		return math.NaN()
	}

	index := q * n
	if index < 0 {
		index = 0
	}
	if index > n-1 {
		index = n - 1
	}
	return data[int(math.Floor(index))]
}

func getSortedCopy(data []float64) []float64 {
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	return sorted
}
