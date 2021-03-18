FROM golang as builder

WORKDIR /app/

# Install dependencies
COPY go* /app/

ENV GO111MODULE=on

# Install tensorflow C-Bindings

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.4.0.tar.gz -O ./libtensorflow.tar.gz
RUN tar -C /usr/local -xzf ./libtensorflow.tar.gz
RUN ldconfig

RUN go mod download
RUN go mod verify

COPY . /app/

# Build project
RUN GOOS=linux go build -ldflags '-w -s' -o /app/main


FROM ubuntu as prod

RUN apt-get update \
    && apt-get install -y wget \
    && wget https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_amd64.deb \
    && dpkg -i dumb-init_1.2.5_amd64.deb \
    && rm dumb-init_1.2.5_amd64.deb \
    && wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.4.0.tar.gz -O ./libtensorflow.tar.gz \
    && tar -C /usr/local -xzf ./libtensorflow.tar.gz \
    && ldconfig \
    && rm ./libtensorflow.tar.gz

COPY --chown=0:0 --from=builder /app/main /app/main
COPY --chown=0:0 --from=builder /app/models/sentiment/ /app/models/sentiment/
COPY --chown=0:0 --from=builder /app/models/technical/ /app/models/technical/


WORKDIR /app
RUN adduser --home /app --no-create-home restricted_user \
    && chown -R restricted_user /app \
    && chmod -R 500 ./main ./models/sentiment/ \
    && umask 777
USER restricted_user
CMD /usr/bin/dumb-init ./main --port=$PORT --token=$TWITTER_TOKEN
