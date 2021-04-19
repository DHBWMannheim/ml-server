[![Go Reference](https://pkg.go.dev/badge/github.com/DHBWMannheim/ml-server.svg)](https://pkg.go.dev/github.com/DHBWMannheim/ml-server)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=DHBWMannheim_ml-server&metric=alert_status)](https://sonarcloud.io/dashboard?id=DHBWMannheim_ml-server)
# ML-Server

## Disclaimer
Alle Beiträge und Artikel auf dieser Seite wurden nach bestem Wissen und mit größtmöglicher Sorgfalt erstellt. Der Herausgeber übernimmt trotz sorgfältiger Überprüfung der zugrundeliegenden Quellen keine Gewähr für den Inhalt des Angebotes. Jede Haftung entstandener Schäden ist ausgeschlossen. Der Herausgeber haftet nicht für eingesandte Manuskripte und Unterlagen.

Die hierin enthaltenen Angaben und Mitteilungen sind ausschließlich zur Information und zum persönlichen Gebrauch bestimmt. Keine der hierin enthaltenen Informationen begründet ein Angebot zum Verkauf oder die Werbung von Angeboten zum Kauf eines Terminkontraktes, eines Wertpapiers oder einer Option.

Haftungsansprüche gegen den Herausgeber, welche sich auf Schäden materieller oder ideeller Art beziehen, die durch die Nutzung oder Nichtnutzung der dargebotenen Informationen bzw. durch die Nutzung fehlerhafter und unvollständiger Informationen verursacht wurden, sind grundsätzlich ausgeschlossen, sofern seitens des Herausgebers kein nachweislich vorsätzliches Verschulden vorliegt.

Alle Beiträge und Artikel auf dieser Seite stellen ausschließlich der Meinung des Herausgebers. Es findet keine Anlageberatung durch den Herausgeber oder die Entwickler des Projektes „Name“ statt. Dieser Beitrag ist eine journalistische Publikation und dient ausschließlich Informations- und Unterhaltungszwecken. Der Beitrag stellt keine Aufforderung zum Kauf oder Verkauf einer Aktie, eines Wertpapiers oder einer sonstigen Anlage dar. Jeder Anleger ist an dieser Stelle dazu aufgefordert, sich seine eigenen Gedanken zu machen, bevor eine Investitionsentscheidung trifft.

Der Kauf von Aktien und anderen Wertpapieren ist mit hohen Risiken bis hin zum Totalverlust behaftet.

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

## Anwendung

Unsere API wird gerade im Rahmen des [Klopapier.exchange Projektes](https://github.com/michael-spengler/klopapier.exchange), welche [hier](https://klopapier.exchange/#/) abgerufen werden kann implementiert.
