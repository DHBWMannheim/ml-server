package util

import (
	"math"
)

func NormalizedSigmoid(x float32) float64 {
	return ((1 / (1 + math.Exp(-float64(x)))) - 0.5) * 2
}

// Calculates the SMA of the given input array with the given window size n
// It calculates the CumSum of the input array and afterwards
// slices it into the final result
func SimpleMovingAverage(input []float64, n int) []float64 {
	cumsum := make([]float64, len(input)+1)
	result := make([]float64, len(input))

	cumsum[0] = 0

	for i, v := range input {
		cumsum[i+1] = cumsum[i] + v
		if i < n-1 {
			result[i] = input[i]
		}
	}

	part1 := cumsum[n:]
	part2 := cumsum[:len(cumsum)-n]

	for i := range part1 {
		// Add offset of first n-1 elements
		result[i+(n-1)] = (part1[i] - part2[i]) / float64(n)
	}

	return result
}
