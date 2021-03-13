# ML-Server

## Getting Started

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Start Flask Server
```bash
FLASK_APP="main.py" flask run
```

## Usage

**Endpoints**

| Endpoint | Result |
| -------- | ------ |
| `GET /sentiment/twitter` | `{"dates": [/*Array of Tweet Creation Dates*/], "weighted_sentiment": [/*Array of weighted and normalized predictions*/]}`|