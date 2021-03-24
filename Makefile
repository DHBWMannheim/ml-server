build:
	go build -ldflags="-s -w" -o bin/ml-server

build-docker:
	docker build .

run-docker:
	docker run -p 5000:5000 -e TWITTER_TOKEN=$(token) -d $(container)