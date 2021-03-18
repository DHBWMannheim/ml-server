package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

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
	flag.Parse()

	if len(*token) == 0 {
		log.Fatal("Twitter API token must be provided!")
	}

	model, err := tf.LoadSavedModel("./models/sentiment/trained", []string{"serve"}, nil)
	if err != nil {
		log.Fatal(err)
	}

	techModel, err := tf.LoadSavedModel("./models/technical/trained", []string{"serve"}, nil)
	if err != nil {
		log.Fatal(err)
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

	sentimentService := sentiment.NewSentimentService(model, tweet)
	technicalService := technical.NewService(techModel)

	http.Handle("/sentiment/twitter", withLogging(sentimentService.TwitterSentiment, l))
	http.Handle("/technical", withLogging(technicalService.TechnicalAnalysis, l))

	if err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil); err != nil {
		log.Fatal(err)
	}
}
