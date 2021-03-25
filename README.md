[![Go Reference](https://pkg.go.dev/badge/github.com/DHBWMannheim/ml-server.svg)](https://pkg.go.dev/github.com/DHBWMannheim/ml-server)
# ML-Server
## Getting Started

1. Install dependencies

```bash
go mod download
```

2. Add datasets and tokens **Only needed, if the model should be newly trained**

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
| `GET /sentiment/twitter` | `[ { "value": <weighted prediction of ML Model>, "date": "2020-11-10T23:00:00Z", "sma": <value of the SMA function at the current position> } ]` |
| `GET /technical/{shareId}` | `[ [ {"value": "<value fetched from yahoo>", "date": "2020-11-10T23:00:00Z"} ], [ {"value": "<value predicted by model>", "date": "2020-11-10T23:00:00Z"} ]]` |

## Todos

- [ ] Bin noch nicht zufrieden mit der gonum/mat Nutzung. Das ist iwie unintuitiv und umständlich
- [x] Hinzufügen der Prediction Dates des nächsten Monats
- [x] mat.Dense brauch feste Rows/Cols, das führt bei dem Laden anderer Aktien zu Fehlern
- [x] techn. Modelle müssen jeden Tag erneuert werden, um valide Predictions zu haben
- [ ] On-Demand Model-Generation
- [ ] Speichern der Modelle in Google Cloud Storage/S3 Bucket?

