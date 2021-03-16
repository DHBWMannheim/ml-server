FROM golang as builder

WORKDIR /app/

# Install dependencies
COPY go* /app/

COPY . /app/

ENV GO111MODULE=on

# Install tensorflow C-Bindings

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.4.0.tar.gz -O ./libtensorflow.tar.gz
RUN tar -C /usr/local -xzf ./libtensorflow.tar.gz
RUN ldconfig

RUN go mod download
RUN go mod verify

# Build project
RUN GOOS=linux go build -ldflags '-w -s' -o /app/main

# CMD /app/main --port=$PORT --token=$TWITTER_TOKEN

FROM ubuntu as prod

RUN apt-get update && apt-get install -y wget

# install tensorflow
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.4.0.tar.gz -O ./libtensorflow.tar.gz
RUN tar -C /usr/local -xzf ./libtensorflow.tar.gz
RUN ldconfig

COPY --from=builder /app/main /main
COPY --from=builder /app/models/sentiment/ /models/sentiment/

CMD /main --port=$PORT --token=$TWITTER_TOKEN
