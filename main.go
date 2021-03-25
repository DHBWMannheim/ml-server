package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/DHBWMannheim/ml-server/cloudstorage"
	"github.com/DHBWMannheim/ml-server/sentiment"
	"github.com/DHBWMannheim/ml-server/technical"
	"github.com/g8rswimmer/go-twitter"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/google/uuid"
	"github.com/hashicorp/go-hclog"
)

type authorize struct {
	Token string
}

func (a *authorize) Add(req *http.Request) {
	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", a.Token))
}

func withLogging(next http.HandlerFunc, logger hclog.Logger) http.HandlerFunc {
	return func(rw http.ResponseWriter, r *http.Request) {
		id := uuid.New()
		rw.Header().Add("x-request-id", id.String())

		start := time.Now()
		next(rw, r)
		logger.Info("serving request", "request-id", id.String(), "path", r.URL.String(), "time", time.Now().Sub(start).Round(time.Millisecond))
	}
}

func main() {

	token := flag.String("token", "", "Bearer token for Twitter API V2")
	port := flag.Int("port", 5000, "port on which to start the server on")
	bucketName := flag.String("bucket", "ml-models-dhbw", "Name of the gcp bucket used to store ML models")
	flag.Parse()

	if len(*token) == 0 {
		log.Fatal("Twitter API token must be provided!")
	}

	tweet := &twitter.Tweet{
		Authorizer: &authorize{
			Token: *token,
		},
		Client: http.DefaultClient,
		Host:   "https://api.twitter.com",
	}

	l := hclog.New(&hclog.LoggerOptions{
		Name:  "main",
		Level: hclog.Info,
		Color: hclog.ColorOption(hclog.AutoColor),
	})

	cs := cloudstorage.NewCloudStorageService(context.Background(), *bucketName)

	// Download initial sentiment model
	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	modelPath := filepath.Join(cwd, "models", "sentiment", "model-sentiment")

	_, err = cs.DownloadModel(context.Background(), "sentiment/model-sentiment.zip", modelPath)
	if err != nil {
		panic(err)
	}

	model, err := tf.LoadSavedModel(modelPath, []string{"serve"}, nil)
	if err != nil {
		log.Fatal(err)
	}

	sentimentService := sentiment.NewSentimentService(model, tweet)
	technicalService := technical.NewService(cs, l)

	// Load ETH-USD initially
	err = technicalService.LoadModel(context.Background(), "ETH-USD")
	if err != nil {
		panic(err)
	}

	http.Handle("/sentiment/twitter", withLogging(sentimentService.TwitterSentiment, l))
	http.Handle("/technical/", withLogging(technicalService.TechnicalAnalysis, l))

	if err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil); err != nil {
		log.Fatal(err)
	}
}
