// Practical mathematical functions used to normalize and transform prediction results
package util

import (
	"archive/zip"
	"context"
	"io"
	"math"
	"os"
	"os/exec"
	"strings"
)

// Sigmoid function based this formula: https://de.wikipedia.org/wiki/Sigmoidfunktion
//
// Transforming value range from [0,1] to [-1,1]
func NormalizedSigmoid(x float32) float64 {
	return ((1 / (1 + math.Exp(-float64(x)))) - 0.5) * 2
}

// Calculates the SMA of the given input array with the given window size n.
//
// It calculates the CumSum of the input array and afterwards
// slices it into the final result.
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

// Calculates the Aroon index based on the closing prices
// and a given period to analyse. Normally the window is
// window=25, but it is customizable.
//
// The output array has the same length as the closing price
// array.
// The missing len(close)-window elements will be inserted at the beginning
// of the result array and initialized with 0
func Aroon(close []float64, window int) (aroon []float64) {

	// aroon[0:window] = 0
	for i := 0; i < window; i++ {
		aroon = append(aroon, 0)
	}

	// aroon[window:len(close)-1] = Aroon()
	for j := window; j < len(close); j++ {
		max, min := 0.0, math.Inf(1)
		var periodMax, periodMin int

		for i := j - window; i < j; i++ {
			price := close[i]
			if price > max {
				max = price
				periodMax = j - i
			}
			if price < min {
				min = price
				periodMin = j - i
			}
		}

		up := (float64((window - periodMax)) / float64(window)) * 100

		down := (float64((window - periodMin)) / float64(window)) * 100

		aroon = append(aroon, up-down)
	}

	return aroon
}

// Removes 0 Values of an Array and returnes the clean array as well
// as the index of the removed items
func RemoveZeroValues(input []float64) ([]float64, []int) {
	var clean []float64
	var removedIndices []int
	for i, e := range input {
		if e == 0 {
			removedIndices = append(removedIndices, i)
			continue
		}
		clean = append(clean, e)
	}

	return clean, removedIndices
}

type Namer interface {
	Name() string
}

// Extracts an archive downloaded from the gcp bucket.
//
// Normally the archive should contain a trained model from Tensorflow; other archives are not verified
func ExtractTfArchive(f Namer, modelPath string) error {
	rc, err := zip.OpenReader(f.Name())
	if err != nil {
		return err
	}

	defer rc.Close()

	for _, zf := range rc.File {
		filePath := modelPath + "/" + zf.Name
		if strings.HasSuffix(zf.Name, "/") {
			err = os.MkdirAll(filePath, 0700)
			if err != nil {
				return err
			}
			continue
		}

		zfc, err := zf.Open()
		if err != nil {
			return err
		}

		ef, err := os.Create(filePath)
		if err != nil {
			return err
		}

		io.Copy(ef, zfc)

		zfc.Close()
		ef.Close()
	}

	return nil
}

func TrainModelLocally(ctx context.Context, shareId string) error {

	cmd := exec.CommandContext(ctx, "python3", "models/technical/train.py", shareId)

	return cmd.Run()
}
