# embedding-service

Local inference service that wraps **BAAI/bge-m3** and exposes a single `/embed` endpoint. Used by both `ingest-processor` (passage embedding) and `query-service` (query embedding). Not deployed to AWS — runs on the developer's laptop and is made reachable from EC2 via an ngrok tunnel managed by `make ngrok-connect`.

---

## What it does

```
POST /embed  { texts: [...], input_type: "passage" | "query" }
     │
     ▼
BGEM3FlagModel.encode()   (CPU, fp16, asyncio.to_thread)
     │
     ├─ dense_vecs   — list of 1024-dim L2-normalised float vectors
     └─ lexical_weights — list of {token_id_str: weight} dicts (sparse)
     │
     ▼
{ model, dense_dim, embeddings: [{ dense: [...], sparse: {...} }] }
```

BGE-M3 is a **symmetric** encoder — `input_type` does not change the model computation. It is retained in the API for caller clarity and logging.

---

## Prerequisites

- Docker Desktop running
- The Docker image must be built (the model is baked in at build time — no download at startup):

```bash
cd embedding-service
docker build -t local/embedding-service:latest .
# Takes ~10 minutes on first build (downloads ~2.2 GB of model weights)
```

---

## Running

```bash
docker start embedding-service
# or, if the container doesn't exist yet:
docker run -d --name embedding-service -p 8000:8000 local/embedding-service:latest
```

Wait for the model to load (~15–30 s on first start):

```bash
curl http://localhost:8000/ready
# {"status":"ready"}  ← ready when this returns 200
```

---

## Connecting to AWS EC2

`ingest-processor` and `query-service` run on EC2 and need to reach this service. Use the project's ngrok tunnel:

```bash
# From the recall/ root — starts local-proxy + ngrok, updates EC2 docker-compose, restarts affected services
make ngrok-connect
```

Run this again after every laptop restart (free-tier ngrok assigns a new URL each session).

---

## API

### `POST /embed`

**Request:**
```json
{
  "texts": ["def handle_sqs(...): ...", "another chunk"],
  "input_type": "passage"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `texts` | `list[str]` | yes | 1 to `MAX_TEXTS_PER_REQUEST` (default 128) |
| `input_type` | `"passage"` \| `"query"` | no | Default `"passage"`. Does not change model output. |

**Response:**
```json
{
  "model": "BAAI/bge-m3",
  "dense_dim": 1024,
  "embeddings": [
    {
      "dense": [0.0123, -0.0456, ...],
      "sparse": {"1037": 0.182, "2843": 0.094}
    }
  ]
}
```

- `embeddings` is index-aligned with `texts`.
- `dense` vectors are L2-normalised (cosine == dot product).
- `sparse` keys are **string** token-ids (JSON requirement); values are positive floats.

### `GET /health`

Always `200` while the process is up.
```json
{"status": "ok", "model_loaded": true}
```

### `GET /ready`

`200` once the model is loaded; `503` while still warming up (~15–30 s after container start).
```json
{"status": "ready"}
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_NAME` | `BAAI/bge-m3` | HuggingFace model ID |
| `USE_FP16` | `true` | fp16 inference — halves memory, faster on CPU AVX |
| `MODEL_BATCH_SIZE` | `12` | Micro-batch size inside one `.encode()` call |
| `MAX_LENGTH` | `8192` | Max token length per text (BGE-M3 supports up to 8192) |
| `MAX_CONCURRENT_INFERENCES` | `2` | Semaphore cap on parallel `encode()` calls |
| `TORCH_NUM_THREADS` | `0` | `0` = let torch decide; pin on noisy multi-tenant hosts |
| `MAX_TEXTS_PER_REQUEST` | `128` | Hard cap on texts per request (returns 422 if exceeded) |
| `LOG_LEVEL` | `INFO` | Python log level |
| `SERVICE_PORT` | `8000` | Uvicorn bind port |

---

## Project structure

```
embedding-service/
├── app/
│   ├── api/
│   │   └── routes.py     # POST /embed, GET /health, GET /ready
│   ├── config.py         # pydantic-settings
│   ├── main.py           # FastAPI app + lifespan (loads/unloads model)
│   ├── model.py          # BGEM3FlagModel singleton + asyncio.to_thread wrapper
│   └── schemas.py        # EmbedRequest, EmbedResponse, SingleEmbedding
├── Dockerfile            # python:3.11-slim + model preloaded at build time
├── requirements.txt      # fastapi, uvicorn, FlagEmbedding, torch, numpy
└── .env.example
```

---

## Design notes

- **Model loaded once in lifespan** — `asyncio.to_thread(load_model)` at startup, never per-request.
- **CPU only** — `devices="cpu"`. No CUDA/MPS code paths (this service runs on an Intel Mac).
- **`asyncio.to_thread` for inference** — `encode()` is blocking CPU work; wrapping it in `to_thread` keeps the event loop unblocked for health checks and concurrent requests.
- **Semaphore serialises inference** — `asyncio.Semaphore(max_concurrent_inferences)` (default 2) prevents multiple simultaneous `encode()` calls from thrashing CPU threads. Torch already uses internal threading, so adding more concurrent callers doesn't help.
- **Model baked into Docker image** — the `Dockerfile` preloads model weights at build time (`RUN python -c "...BGEM3FlagModel(...)"`) so cold start is disk load (~15 s), not a 2 GB download.
- **Not deployed to AWS** — this service is local-only. `ingest-processor` fails fast at startup if `EMBEDDING_SERVICE_URL` is unset or unreachable.
