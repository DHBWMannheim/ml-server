// Provides an http.HandlerFunc which predicts future prices based on a technical Model
package technical

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"github.com/DHBWMannheim/ml-server/cloudstorage"
	"github.com/DHBWMannheim/ml-server/util"
	"github.com/hashicorp/go-hclog"
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
	// http.Handler to execute Code for technical analysis
	//
	// It accepts every valid quoteId for https://finance.yahoo.com and
	// train a model and predict future stock values
	TechnicalAnalysis(http.ResponseWriter, *http.Request)
	// Loads a specific model either if present in the current fs
	// or from a remote location, which can be specified by the -bucket flag.
	//
	// In case no model is present, if the value is a valid quoteId from
	// https://finance.yahoo.com, a new model is trained and provided in the
	// remote location
	LoadModel(context.Context, string) error
}

type service struct {
	currentModel string
	storage      cloudstorage.Storage
	model        *tf.SavedModel
	l            hclog.Logger
}

func NewService(storage cloudstorage.Storage, l hclog.Logger) Service {
	return &service{storage: storage, l: l, currentModel: "ETH-USD"}
}

type predictionResult struct {
	Value float32 `json:"value,omitempty"`
	Date  string  `json:"date,omitempty"`
}

func (s *service) LoadModel(ctx context.Context, shareId string) error {

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}

	modelPath := filepath.Join(cwd, "models", "technical", fmt.Sprintf("model-%s", shareId))
	s.l.Info("attempt to load model from", "path", modelPath)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		s.l.Info("model not present locally, try getting from remote")
		_, err := s.storage.DownloadModel(ctx, fmt.Sprintf("technical/model-%s.zip", shareId), modelPath)

		if errors.Is(err, storage.ErrObjectNotExist) {
			s.l.Info("model not present remotly, triggering training")
			if err := util.TrainModelLocally(ctx, shareId); err != nil {
				return err
			}
			return errors.New("Intentional breakpoint")
		}

		if err != nil {
			return err
		}
	}

	model, err := tf.LoadSavedModel(modelPath, []string{"serve"}, nil)
	if err != nil {
		return err
	}

	s.model = model
	s.currentModel = shareId
	return nil
}

func (s *service) TechnicalAnalysis(w http.ResponseWriter, r *http.Request) {

	shareId := "ETH-USD"
	daysInFuture := 30
	params := r.URL.Query()

	if share := strings.TrimPrefix(r.URL.Path, "/technical/"); !strings.Contains(share, "/") && len(share) > 0 {
		// Make sure parameter is uppercased, to ensure correct naming and mapping from yahoo
		shareId = strings.ToUpper(share)
	}

	if s.currentModel != shareId {
		// TODO: make model unbound to service struct to ensure concurrency
		if err := s.LoadModel(r.Context(), shareId); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	if days, ok := params["days"]; ok {
		parsed, err := strconv.ParseInt(days[0], 10, 64)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	start := time.Now().Format("2006-01-02")
	end := time.Now().AddDate(0, 0, -100).Format("2006-01-02")

	quotes, err := quote.NewQuoteFromYahoo(shareId, end, start, quote.Daily, true)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	input, dates := normalizeYahooData(quotes)

	for i := 0; i < daysInFuture; i++ {

		rawMatrix := mat.NewDense(len(input), 11, nil)
		rawMatrix.SetCol(0, input)

		rawMatrix = generateIndicators(rawMatrix, dates)

		reshaped := mat.NewDense(31, 11, nil)

		for r := 0; r < 31; r++ {
			reshaped.SetRow(31-r-1, rawMatrix.RawRowView(len(input)-r-1))
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
		input = append(input, rr.At(0, 0))
		dates = append(dates, dates[len(dates)-1].AddDate(0, 0, 1))
	}

	var result = make([][]*predictionResult, 2)

	historical, predicted := input[:len(input)-daysInFuture], input[len(input)-daysInFuture:]

	for i, h := range historical {
		result[0] = append(result[0], &predictionResult{
			Value: float32(h),
			Date:  dates[i].Format(time.RFC3339),
		})
	}

	for i, p := range predicted {
		result[1] = append(result[1], &predictionResult{
			Value: float32(p),
			Date:  dates[len(historical)+i].Format(time.RFC3339),
		})
	}

	w.Header().Add("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func normalizeYahooData(in quote.Quote) ([]float64, []time.Time) {
	input, ri := util.RemoveZeroValues(in.Close)

	var dates []time.Time

dateLoop:
	for di, d := range in.Date {
		for _, i := range ri {
			if di == i {
				continue dateLoop
			}
		}
		dates = append(dates, d)
	}
	return input, dates
}

func generateIndicators(input *mat.Dense, dates []time.Time) *mat.Dense {

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

	for i, t := range dates {
		input.Set(i, 8, float64(t.Day()))
		input.Set(i, 9, float64(t.Weekday()))
		input.Set(i, 10, float64(t.Month()))
	}

	return input
}

func matToMultiArray(input mat.Matrix) [][]float32 {
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
