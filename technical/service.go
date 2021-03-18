// Provides an http.HandlerFunc which predicts future prices based on a technical Model
package technical

import (
	"fmt"
	"net/http"
	"time"

	"github.com/DHBWMannheim/ml-server/util"
	"github.com/pa-m/sklearn/preprocessing"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/markcheno/go-quote"
	"github.com/markcheno/go-talib"
	"gonum.org/v1/gonum/mat"
)

const (
	tfInput  = "serving_default_lstm_input"
	tfOutput = "StatefulPartitionedCall"
)

type Service interface {
	TechnicalAnalysis(http.ResponseWriter, *http.Request)
}

type service struct {
	model *tf.SavedModel
}

func NewService(model *tf.SavedModel) Service {
	return &service{model}
}

func (s *service) TechnicalAnalysis(w http.ResponseWriter, r *http.Request) {

	shareId := "ETH-USD"
	params := r.URL.Query()

	if share, ok := params["share"]; ok {
		if s := share[0]; len(s) > 0 {
			shareId = s
		}
	}

	start := time.Now().Format("2006-01-02")
	end := time.Now().AddDate(0, 0, -100).Format("2006-01-02")

	quotes, err := quote.NewQuoteFromYahoo(shareId, end, start, quote.Daily, true)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	input, ri := util.RemoveZeroValues(quotes.Close)

	var dates []time.Time

	for _, i := range ri {
		for di, d := range quotes.Date {
			if di == i {
				continue
			}
			dates = append(dates, d)
		}
	}

	rawMatrix := mat.NewDense(100, 11, nil)
	rawMatrix.SetCol(0, input)

	rawMatrix = generateIndicators(rawMatrix)
	rawMatrix = parseDatesToIndicators(rawMatrix, dates)

	reshaped := mat.NewDense(31, 11, nil)

	for r := 0; r < 31; r++ {
		reshaped.SetRow(31-r-1, rawMatrix.RawRowView(100-r-1))
	}

	scaler := preprocessing.NewMinMaxScaler([]float64{0, 1})
	priceScaler := preprocessing.NewMinMaxScaler([]float64{0, 1})

	scaled, _ := scaler.FitTransform(reshaped, nil)

	priceScaler.FitTransform(reshaped.ColView(0), nil)

	modelInput := matToMultiArray(scaled)

	tensor, _ := tf.NewTensor([][][]float32{modelInput})

	results, err := s.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			s.model.Graph.Operation(tfInput).Output(0): tensor,
		},
		[]tf.Output{
			s.model.Graph.Operation(tfOutput).Output(0),
		}, nil)

	if err != nil {
		http.Error(w, "could not predict stock", http.StatusInternalServerError)
		return
	}

	f := results[0].Value().([][]float32)

	p := mat.NewDense(1, 1, []float64{float64(f[0][0])})

	rr, _ := priceScaler.InverseTransform(p, nil)

	fmt.Fprintf(w, "ETH-USD tomorrow %.2f", rr.At(0, 0))
}

func matToMultiArray(input *mat.Dense) [][]float32 {
	var modelInput [][]float32

	for r := 0; r < 31; r++ {
		var rowData []float32
		for c := 0; c < 11; c++ {
			rowData = append(rowData, float32(input.At(r, c)))
		}
		modelInput = append(modelInput, rowData)
	}

	return modelInput
}

func generateIndicators(input *mat.Dense) *mat.Dense {

	prices := make([]float64, input.RawMatrix().Rows)
	mat.Col(prices, 0, input)
	input.SetCol(1, talib.Kama(prices, 10))

	input.SetCol(2, talib.Ppo(prices, 10, 20, talib.EMA))

	input.SetCol(3, talib.Roc(prices, 12))

	macd, _, _ := talib.Macd(prices, 12, 20, 9)
	input.SetCol(4, macd)

	input.SetCol(5, talib.Rsi(prices, 14))

	input.SetCol(6, util.Aroon(prices, 20))

	_, bbands, _ := talib.BBands(prices, 20, 2, 2, talib.SMA)
	input.SetCol(7, bbands)

	return input
}

func parseDatesToIndicators(indicators *mat.Dense, dates []time.Time) *mat.Dense {

	for i, t := range dates {
		indicators.Set(i, 8, float64(t.Day()))
		indicators.Set(i, 9, float64(t.Weekday()))
		indicators.Set(i, 10, float64(t.Month()))
	}

	return indicators
}
