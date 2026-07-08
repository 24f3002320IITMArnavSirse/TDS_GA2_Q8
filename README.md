# Local LLM Structured-Output Service

FastAPI service that extracts structured invoice fields (`vendor`, `amount`, `currency`, `date`) from free-form invoice text using a hybrid architecture: Ollama LLM with deterministic parser fallback.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

**POST /extract**

```json
{"text": "<invoice text>"}
```

Response:

```json
{
  "vendor": "Acme-XXXX Industries Ltd.",
  "amount": 1234.56,
  "currency": "USD",
  "date": "2026-04-19"
}
```

## Environment variables

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` |
| `OLLAMA_TIMEOUT` | `5` |
| `PORT` | `8000` |

## Tests

```bash
pytest -v
```

## Docker

```bash
docker build -t invoice-extractor .
docker run -p 8000:8000 invoice-extractor
```

## Deploy to Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Connect the repo, set **Environment** to Docker (or Python with build command `pip install -r requirements.txt` and start command `uvicorn main:app --host 0.0.0.0 --port $PORT`).
4. Set `PORT` (Render sets this automatically).
5. Deploy and use `https://<your-service>.onrender.com/extract`.
