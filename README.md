# ML-Server

## Getting Started

1. Install dependencies

```bash
go mod download
```

2. Add datasets and tokens

- add `.twitter_keys.yaml` to /data/
- add `tweets.csv` to /data/ - Download [here](https://www.dropbox.com/s/ur7pw797mgcc1wr/tweets.csv?dl=0)

3. Start Go Server

```bash
go run main.go --token="<Twitter API Token>" [--port=5000]
```

The `--port` flag is optional and has a default value of 5000

## Usage

**Endpoints**

| Endpoint                 | Result                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `GET /sentiment/twitter` | `{"dates": [/*Array of Tweet Creation Dates*/], "weighted_sentiment": [/*Array of weighted and normalized predictions*/]}` |
