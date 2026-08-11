#!/usr/bin/env python
"""Scores retrieval against the labelled set in evals/retrieval_questions.jsonl.

Retrieval only - no generation, so this costs query embeddings rather than chat tokens,
and it measures the layer that silently degrades when chunking, top_k or the embedding
model changes.

Usage:
    python scripts/ingest.py --seasons 2022 2023 --skip-wikipedia
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 40 --questions evals/retrieval_questions.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore

from f1_trivia_rag.config import settings
from f1_trivia_rag.eval.retrieval import evaluate, load_questions
from f1_trivia_rag.rag.query_engine import (
    MAX_SEASON_TOP_K,
    IndexUnavailableError,
    SeasonAwareRetriever,
    _configure_llama_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "retrieval_questions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--k",
        type=int,
        default=MAX_SEASON_TOP_K,
        help="Cut retrieved results off at k before scoring (default: MAX_SEASON_TOP_K).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    _configure_llama_index()

    if not settings.chroma_persist_dir.exists():
        raise IndexUnavailableError(
            f"No index at {settings.chroma_persist_dir}. Run scripts/ingest.py first."
        )

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = client.get_or_create_collection(settings.chroma_collection)
    if collection.count() == 0:
        raise IndexUnavailableError(
            f"Collection '{settings.chroma_collection}' is empty. Run scripts/ingest.py first."
        )

    index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))
    retriever = SeasonAwareRetriever(index, collection)

    def retrieve(question: str) -> list[str]:
        nodes = retriever.retrieve(QueryBundle(question))
        # Distinct source documents, best-ranked first: a document split into several
        # nodes should count once, or recall would depend on the chunk size.
        ordered: list[str] = []
        for node in nodes:
            source_id = node.metadata.get("source_id")
            if source_id and source_id not in ordered:
                ordered.append(source_id)
        return ordered

    def season_source_ids(season: str) -> list[str]:
        rows = collection.get(where={"season": season}, include=["metadatas"])
        return sorted({m.get("source_id") for m in rows["metadatas"] if m.get("source_id")})

    questions = load_questions(args.questions)
    report = evaluate(questions, retrieve, k=args.k, season_source_ids=season_source_ids)

    print(f"corpus         {collection.count()} nodes in '{settings.chroma_collection}'")
    print(report.format())

    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
