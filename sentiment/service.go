// Provides an http.HandlerFunc which returns a seniment analysis based on tweets
package sentiment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/DHBWMannheim/ml-server/util"
	"github.com/g8rswimmer/go-twitter"
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

const tensorflowInput = "serving_default_text_vectorization_input"
const tensorflowOutput = "StatefulPartitionedCall"

var fieldOpts = &twitter.TweetFieldOptions{
	TweetFields: []twitter.TweetField{twitter.TweetFieldCreatedAt, twitter.TweetFieldPublicMetrics},
}

type twitterResult struct {
	Text  string
	Likes int
	Date  time.Time
}

type twitterResults []*twitterResult

// Transforms the `Text` value of the Tweet into a Tensor to be interpreted by tensorflow
func (t *twitterResults) ToTensor() (*tf.Tensor, error) {
	var input [][1]string
	for _, t := range *t {
		input = append(input, [1]string{t.Text})
	}
	return tf.NewTensor(input)
}

type predictionResult struct {
	// Weighted prediction result from tensorflow
	Value float32 `json:"value,omitempty"`
	// Creation date of the predicted Tweet
	Date string `json:"date,omitempty"`
	// Simple moving average of the prediction
	SMA float32 `json:"sma,omitempty"`
}

type sentimentService struct {
	model *tf.SavedModel
	tweet *twitter.Tweet
}

type Service interface {
	// TwitterSentiment function can be used with the `net/http` package as a http.HandlerFunc.
	// It collects the most recent tweets from twitter, with tweets matching the provides query.
	// It fetches at most the last 100 Tweets but must fetch a minimum of 10.
	//
	// Default Values (can be overriden py providing URL-Params):
	// 		"twitter_query": "ether OR eth OR ethereum OR cryptocurrency"
	//		"tweet_count": 10 <= tweet_count <= 100
	TwitterSentiment(http.ResponseWriter, *http.Request)
}

// Creates a new sentiment service with a given tensorflow model and a twitter api client
func NewSentimentService(model *tf.SavedModel, tweet *twitter.Tweet) Service {
	return &sentimentService{model, tweet}
}

// Implements the http.Handler interface and handles incoming HTTP requests
func (s *sentimentService) TwitterSentiment(w http.ResponseWriter, req *http.Request) {

	if req.Method != http.MethodGet {
		http.NotFound(w, req)
		return
	}

	twitterQuery := "ether OR eth OR ethereum OR cryptocurrency"
	tweetCount := 100

	params := req.URL.Query()
	if query, ok := params["tweet_count"]; ok {
		parsed, err := strconv.ParseInt(query[0], 10, 8)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if parsed < 10 || parsed > 100 {
			err = errors.New(fmt.Sprintf("tweet_count %d is not between 10 and 100", parsed))
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		tweetCount = int(parsed)
	}

	if query, ok := params["twitter_query"]; ok {
		parsed := query[0]

		keywordCount := len(strings.Split(parsed, "OR"))

		if keywordCount >= 10 || keywordCount == 0 {
			err := errors.New(fmt.Sprintf("twitter_query has %d keywords, but must be betweent 1 and 10", keywordCount))
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		twitterQuery = parsed
	}

	searchOpts := &twitter.TweetRecentSearchOptions{
		MaxResult: tweetCount,
	}

	tweets, err := s.tweet.RecentSearch(context.Background(), fmt.Sprintf("(%s) -bot -app -is:retweet is:verified lang:en", twitterQuery), *searchOpts, *fieldOpts)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var tweetResults twitterResults

	for _, lookup := range tweets.LookUps {

		date, err := time.Parse(time.RFC3339Nano, lookup.Tweet.CreatedAt)

		if err != nil {
			http.Error(w, fmt.Sprintf("could not parse date %s", lookup.Tweet.CreatedAt), http.StatusInternalServerError)
			return
		}

		tweetResults = append(tweetResults, &twitterResult{
			Text:  lookup.Tweet.Text,
			Likes: lookup.Tweet.PublicMetrics.Likes,
			Date:  date,
		})
	}

	sort.Slice(tweetResults, func(i, j int) bool {
		return tweetResults[i].Date.Before(tweetResults[j].Date)
	})

	input, err := tweetResults.ToTensor()

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	results, err := s.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			s.model.Graph.Operation(tensorflowInput).Output(0): input,
		}, []tf.Output{
			s.model.Graph.Operation(tensorflowOutput).Output(0),
		}, nil)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	res := make([]*predictionResult, tweetCount)
	weights := make([]float64, tweetCount)

	for i, pred := range results[0].Value().([][]float32) {
		weight := tweetResults[i].Likes + 1
		weightedSentiment := util.NormalizedSigmoid(float32(weight) * pred[0])

		weights[i] = weightedSentiment
		res[i] = &predictionResult{
			Value: float32(weightedSentiment),
			Date:  tweetResults[i].Date.Format(time.RFC3339Nano),
		}
	}

	weights = util.SimpleMovingAverage(weights, 10)

	for i, w := range weights {
		res[i].SMA = float32(w)
	}

	w.Header().Add("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}
