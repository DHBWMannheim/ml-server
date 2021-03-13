from flask import Flask, jsonify, request, abort
import tensorflow as tf
from tensorflow import keras
from searchtweets import load_credentials, gen_request_parameters, collect_results
from datetime import datetime
import math


app = Flask(__name__)

search_args = load_credentials("./.twitter_keys.yaml", yaml_key="search_tweets_v2")

@tf.keras.utils.register_keras_serializable()
def normalize_data(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase, '@(\w*)|(\\n)|(https:\/\/t\.co[\w\/]*)', '')

model = keras.models.load_model("./sentiment")

def normalized_sigmoid(x):
    return ((1 / (1 + math.exp(-x))) - 0.5) * 2

@app.route("/sentiment/twitter", methods=["GET"])
def sentiment_analysis():
    max_tweets = int(request.args['tweet_count']) if request.args and 'tweet_count' in request.args else 100
    twitter_query: str = request.args['twitter_query'] if request.args and 'twitter_query' in request.args else "ether OR eth OR ethereum OR cryptocurrency"

    if max_tweets < 10 or max_tweets > 100:
        abort(400, 'max_tweets not between 10 and 100')

    query = gen_request_parameters(
        "({}) -bot -app -is:retweet is:verified lang:en".format(twitter_query),
        tweet_fields="id,created_at,text,public_metrics",
        results_per_call=max_tweets)
    
    tweets = list(reversed(collect_results(query, max_tweets=max_tweets, result_stream_args=search_args)))

    # Remove first summary element
    tweets.pop(0)

    create_dates = []
    tweet_texts = []
    weighted_sentiment = []

    for tweet in tweets:
        tweet_texts.append(tweet['text'])
        
        utc_time = datetime.strptime(tweet['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
        epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
        create_dates.append(epoch_time)

    sentiment = model.predict(tweet_texts).flatten()

    for i in range(len(tweets)):
        weight = tweets[i]['public_metrics']['like_count'] + 1
        weighted_sentiment.append(normalized_sigmoid(weight * sentiment[i]))

    return jsonify({
        'dates': create_dates,
        'weighted_sentiment': weighted_sentiment
    })
