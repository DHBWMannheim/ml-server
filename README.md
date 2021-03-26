[![Go Reference](https://pkg.go.dev/badge/github.com/DHBWMannheim/ml-server.svg)](https://pkg.go.dev/github.com/DHBWMannheim/ml-server)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=DHBWMannheim_ml-server&metric=alert_status)](https://sonarcloud.io/dashboard?id=DHBWMannheim_ml-server)
# ML-Server
## Getting Started

1. Install dependencies

```bash
go mod download
```

2. Add datasets and tokens **Only needed, if the model should be newly trained**

- add `googlecloud.json` to /data/
- add `tweets.csv` to /data/ - Download [here](https://www.dropbox.com/s/ur7pw797mgcc1wr/tweets.csv?dl=0)

3. Start Go Server

```bash
go run main.go --token="<Twitter API Token>" [--port=5000 --bucket="ml-models-dhbw"]
```

The `--port` flag is optional and has a default value of 5000.
The `--bucket` flag is optional and has a default value of "ml-models-dhbw".

## Usage

**Endpoints**

| Endpoint                 | Result                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `GET /sentiment/twitter` | `[ { "value": <weighted prediction of ML Model>, "date": "2020-11-10T23:00:00Z", "sma": <value of the SMA function at the current position> } ]` |
| `GET /technical/{shareId}` | `[ [ {"value": "<value fetched from yahoo>", "date": "2020-11-10T23:00:00Z"} ], [ {"value": "<value predicted by model>", "date": "2020-11-10T23:00:00Z"} ]]` |

## Todos

- [x] Speichern der Modelle in Google Cloud Storage/S3 Bucket?
- [ ] Bug im Sentiment Model beheben
- [ ] On-Demand Model-Generation

