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

func ExampleRemoveZeroValues() {
	clean, ri := util.RemoveZeroValues([]float64{1, 2, 5, 0, 4, 1, 5, 0, 3})

	fmt.Println(clean, ri)
	//Output: [1 2 5 4 1 5 3] [3 7]
}

func TestSimpleMovingAverage(t *testing.T) {
	res := util.SimpleMovingAverage([]float64{1, 2, 3, 4}, 2)

	t.Log(res)

}

func TestAroon(t *testing.T) {
	aroon := util.Aroon([]float64{1, 2, 3, 4, 5, 6, 32, 15, 16, 7, 8, 6, 4, 4, 67, 8, 6, 5, 32, 22, 4, 5, 67, 5, 3}, 20)

	t.Log(aroon)
}

func TestRemoveZeroValues(t *testing.T) {
	testCases := []struct {
		desc string
		in   []float64
		out  int
	}{
		{
			desc: "should remove two 0 values",
			in:   []float64{1, 2, 5, 0, 4, 1, 5, 0, 3},
			out:  2,
		},
		{
			desc: "should return empty array for zero array",
			in:   []float64{0, 0, 0, 0},
			out:  4,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			clean, ri := util.RemoveZeroValues(tC.in)

			if len(ri) != tC.out {
				t.Fatalf("expected to remove 2 items from the array, but removed %d insted", len(ri))
			}

			if len(clean) != len(tC.in)-tC.out {
				t.Fatalf("the output array has not the right length after cleaning. exptected %d but got %d", len(tC.in)-tC.out, len(clean))
			}
		})
	}
}

func TestNormalizedSigmoid(t *testing.T) {
	val := util.NormalizedSigmoid(2)
	if math.Round((val/2+0.5)*100)/100 != float64(0.88) {
		t.Fatalf("calculated value was wrong %v", val/2+0.5)
	}
}
