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

## Notes

```go
package main

import (
	"math"

	"github.com/markcheno/go-quote"
	_ "github.com/markcheno/go-talib"
)

func main() {

	eth, _ := quote.NewQuoteFromYahoo("ETH-USD", "2000-09-01", "2021-03-18", quote.Daily, true)

	// Entfernt die 0 die nicht in den datensatz gehören
	var cs []float64
	for _, e := range eth.Close {
		if e == 0 {
			continue
		}
		cs = append(cs, e)
	}

	var aroon []float64

	n := 20

	for j := n; j < len(cs); j++ {
		var max float64
		min := math.Inf(1)
		var periodMax, periodMin int

		for i := j - n; i < j; i++ {
			price := cs[i]
			if price > max {
				max = price
				periodMax = j - i
			}
			if price < min {
				min = price
				periodMin = j - i
			}
		}

		up := (float64((n - periodMax) / n)) * 100

		down := (float64((n - periodMin) / n)) * 100

		aroon = append(aroon, up-down)
	}

}
```

* Erste 30 Einträge (wegen NaN) rausschneiden
* erstes Array bei MACD
* Zum Predicten die letzten 100 Tage laden
* talib.EMA bei Ppo
* talib.SMA bei BBand
* bei BBands das 2. Array

