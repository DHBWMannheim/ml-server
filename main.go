package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/DHBWMannheim/ml-server/seniment"
	"github.com/g8rswimmer/go-twitter"
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type authorize struct {
	Token string
}

func (a *authorize) Add(req *http.Request) {
	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", a.Token))
}

func main() {

	token := flag.String("token", "", "Bearer token for Twitter API V2")
	port := flag.Int("port", 5000, "port on which to start the server on")
	flag.Parse()

	if len(*token) == 0 {
		panic("Twitter API token must be provided!")
	}

	model, err := tf.LoadSavedModel("./models/sentiment/trained", []string{"serve"}, nil)
	if err != nil {
		panic(err)
	}

	tweet := &twitter.Tweet{
		Authorizer: &authorize{
			Token: *token,
		},
		Client: http.DefaultClient,
		Host:   "https://api.twitter.com",
	}

	service := seniment.NewSentimentService(model, tweet)

	http.Handle("/sentiment/twitter", http.HandlerFunc(service.TwitterSentiment))

	if err := http.ListenAndServe(fmt.Sprintf(":%d", *port), nil); err != nil {
		log.Fatal(err)
	}
}
