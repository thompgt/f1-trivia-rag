# F1 Trivia RAG

A retrieval-augmented chatbot that answers Formula 1 historical stats and trivia questions,
grounded in Ergast race-result data and Wikipedia race reports (with citations back to source).

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-1B1B1D?style=for-the-badge)
![Chroma](https://img.shields.io/badge/Chroma-FF6B6B?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

## Why RAG

LLMs are unreliable at precise sports statistics from memory alone. This project retrieves
grounded facts (race results, standings, race reports) before generating an answer, so
responses cite where the number came from instead of hallucinating it.

## Data sources

- **Ergast API** — structured historical race results, standings, driver/constructor data.
- **Wikipedia** — race report prose (narrative context Ergast doesn't have).
- **FastF1** (optional) — session-level telemetry facts for more recent seasons.

## Model provider

Embeddings and chat generation use Google Gemini (`gemini-2.5-flash` +
`models/gemini-embedding-001`) via `llama-index-llms-gemini` / `llama-index-embeddings-gemini`.

## Project layout

```
src/f1_trivia_rag/
  config.py            Settings (env vars via pydantic-settings)
  ingestion/            Pulls raw data and normalizes it into Documents
    ergast.py
    wikipedia_source.py
    fastf1_source.py
  rag/
    build_index.py      Builds/persists the Chroma vector index
    query_engine.py     Loads the index and answers queries with citations
  api/
    main.py              FastAPI app exposing POST /chat
scripts/
  ingest.py              CLI: fetch data + (re)build the index
tests/
  test_config.py
data/                    raw/ and processed/ ingestion output (gitignored)
storage/                 persisted Chroma index (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env  # then fill in GEMINI_API_KEY
```

## Usage

```bash
# 1. Ingest data and build the vector index
python scripts/ingest.py --seasons 2018 2019 2020 2021 2022 2023

# 2. Run the API
uvicorn f1_trivia_rag.api.main:app --reload
```

Then query it:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many times has a driver won from pole at Monaco since 2018?"}'
```

## Status

Initial scaffolding. Ingestion sources, index build, and API are stubbed with clear
TODOs to fill in as the project progresses.
