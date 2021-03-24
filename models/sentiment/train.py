
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import feature_column

from google.cloud import storage
import shutil

dataframe = pd.read_csv("data/tweets.csv", usecols=[0, 5], names=["target", "text"])

dataframe = shuffle(dataframe)

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'training tweets')
print(len(val), 'validation tweets')
print(len(test), 'test tweets')

def df_to_dataset(dataframe, batch_size):
  dataframe = dataframe.copy()
  texts = dataframe.pop('text')
  labels = dataframe.pop('target')
  return tf.data.Dataset.from_tensor_slices((texts, labels)).batch(batch_size)

batch_size = 320

raw_train_ds = df_to_dataset(train, batch_size)
raw_val_ds = df_to_dataset(val, batch_size)
raw_test_ds = df_to_dataset(test, batch_size)

@tf.keras.utils.register_keras_serializable()
def normalize_data(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase, '@(\w*)|(\\n)|(https:\/\/t\.co[\w\/]*)', '')

max_features = 10000
sequence_length = 56

vectorize_layer = TextVectorization(
    standardize=normalize_data,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), int(label / 4)

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


embedding_dim = 32

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)
])

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

sentiment_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

examples = [
  "Today is a great day!",
  "This sentence is rather neutral",
  "This show is terrible!"
]

sentiment_model.predict(examples)

# save model to temp folder
model.save("models/sentiment/temp/model-sentiment")

# delete empty assets and zip model
shutil.rmtree("models/sentiment/temp/model-sentiment/assets")
shutil.make_archive("models/sentiment/temp/model-sentiment", "zip", "models/sentiment/temp/model-sentiment")

# authenticate in google cloud
storage_client = storage.Client.from_service_account_json("data/googlecloud.json")

# upload zipped model to bucket
bucket_name = "ml-models-dhbw"
source_file_name = "models/sentiment/temp/model-sentiment.zip"
destination_blob_name = "sentiment/model-sentiment.zip"

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)

blob.upload_from_filename(source_file_name)

# remove temp content
shutil.rmtree("models/sentiment/temp/model-sentiment")
os.remove("models/sentiment/temp/model-sentiment.zip")

print(
    "File {} uploaded to {}.".format(
        source_file_name, destination_blob_name
    )
)
