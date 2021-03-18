package util_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/DHBWMannheim/ml-server/util"
)

func ExampleSimpleMovingAverage() {
	res := util.SimpleMovingAverage([]float64{1, 2, 3, 4}, 2)

	fmt.Println(res)
	//Output: [1 1.5 2.5 3.5]
}

func TestSimpleMovingAverage(t *testing.T) {
	res := util.SimpleMovingAverage([]float64{1, 2, 3, 4}, 2)

	t.Log(res)

}

func TestNormalizedSigmoid(t *testing.T) {
	val := util.NormalizedSigmoid(2)
	if math.Round((val/2+0.5)*100)/100 != float64(0.88) {
		t.Fatalf("calculated value was wrong %v", val/2+0.5)
	}
}
