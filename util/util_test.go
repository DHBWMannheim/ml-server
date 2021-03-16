package util

import (
	"math"
	"testing"
)

func TestSimpleMovingAverage(t *testing.T) {
	res := SimpleMovingAverage([]float64{1, 2, 3, 4}, 2)

	t.Log(res)

}

func TestNormalizedSigmoid(t *testing.T) {
	val := NormalizedSigmoid(2)
	if math.Round((val/2+0.5)*100)/100 != float64(0.88) {
		t.Fatalf("calculated value was wrong %v", val/2+0.5)
	}
}
