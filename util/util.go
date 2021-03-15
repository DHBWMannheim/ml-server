package util

import (
	"math"
)

func NormalizedSigmoid(x float32) float64 {
	return ((1 / (1 + math.Exp(-float64(x)))) - 0.5) * 2
}
