# themis

## Batch inference REST API

Install and serve a trained DeBERTa model (a training output directory or its
`model/` subdirectory):

```sh
uv sync --extra api --extra training --extra cpu
THEMIS_MODEL_DIRECTORY=/path/to/training-output uv run uvicorn themis.api:create_app --factory --host 127.0.0.1 --port 8000
```

For CUDA, replace `--extra cpu` with `--extra cuda`. Device selection defaults
to CUDA when available; set `THEMIS_DEVICE=cpu` to override it. The model is
loaded once at startup per worker, so use one worker per GPU.

```sh
curl http://127.0.0.1:8000/v1/inference \
  -H 'Content-Type: application/json' \
  -d '["hello everyone", "good game"]'
```

The response is an array in input order. Each item contains `category`,
`action`, `category_probabilities`, and `action_probabilities`; the probability
maps contain a score for every trained label. Category and action scores are
separately normalized. Duplicate chats retain separate results.

Requests accept up to 256 strings of 1–16,384 characters each. An empty array
returns `[]`; invalid requests return HTTP 422. Model token limits still apply
and longer chats are truncated by the tokenizer. Inference requests share the
model serially, with an internal batch size of 32.

Interactive docs: `http://127.0.0.1:8000/docs`. The OpenAPI definition is checked
in as [`openapi.json`](openapi.json) and served at `/openapi.json`.

Python callers can use `themis.api.create_app(model_directory=..., device=...,
max_length=..., batch_size=...)` to configure the app directly.
