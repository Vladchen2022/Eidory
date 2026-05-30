# Eidory

Eidory v0.1 is a local-first semantic image library for personal use.

It indexes image folders in place, generates local thumbnails, stores metadata in SQLite, and searches image embeddings with an in-memory NumPy index. It does not move, copy, delete, or modify source images.

## Scope

Implemented:

- In-place folder indexing for JPG, JPEG, PNG, WebP, MP4, MOV, M4V, AVI, MKV, and WebM
- Thumbnail cache under `~/Library/Application Support/Eidory/thumbnails`
- SQLite metadata store under `~/Library/Application Support/Eidory/eidory.sqlite3`
- Manual tags, favorite flag, and notes
- Jina CLIP v2 embedding provider for local semantic search
- Similar-image search using stored image embeddings
- Color search and stacked search filters
- AI inspiration projects through local LM Studio: generate visual semantic probes, select up to 7, and search mixed reference results
- LLM settings for LM Studio, OpenAI API, DeepSeek API, Ollama, and generic OpenAI-compatible endpoints
- Background embedding worker with pause/resume/stop
- Video thumbnail generation and preview playback
- Nested collection folders with manual image assignment
- PySide6 desktop UI using Qt Model/View for the image grid

Intentionally not included:

- AI chat recommendations
- AI tag generation
- Cloud sync
- Real-time file watching
- GIF, HEIC, PSD, PDF, and other asset formats
- Signed distribution and auto-update

## Run From Source

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
eidory
```

The first semantic indexing run downloads `jinaai/jina-clip-v2` from Hugging Face. That model requires `trust_remote_code=True` and is licensed `cc-by-nc-4.0`; this project treats it as a personal-use default.

The AI inspiration panel defaults to a local LM Studio OpenAI-compatible server at `http://localhost:1234/v1`. It does not require an OpenAI API key. Advanced users can override `llm.lmstudio.base_url` and `llm.lmstudio.model` in `app_settings`.

## Build macOS App

```bash
./scripts/build_macos_app.sh
open dist/Eidory.app
```

This creates a local unsigned `.app` for personal use. macOS may require right-clicking the app and choosing Open the first time.

## Run Tests

```bash
python -m py_compile $(find src tests -name '*.py')
python -m pytest -q
```

The tests use fake embeddings and do not download the real model.

## Maintenance Boundary

The UI is already broad enough for this stage. Avoid adding more Eagle/Billfish-style
management features before splitting `src/eidory/ui/main_window.py` into smaller
controllers or panels. New AI work should first reuse the existing embedding index
or add a clearly isolated service with fake-provider tests.
