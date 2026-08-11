"""Offline tests for index construction: no embeddings, no network."""

import uuid

import chromadb

from f1_trivia_rag.ingestion.common import RawDocument
from f1_trivia_rag.rag.build_index import reset_collection, to_llama_documents


def test_reset_collection_creates_when_absent():
    client = chromadb.EphemeralClient()
    collection = reset_collection(client, f"absent-{uuid.uuid4().hex[:8]}")
    assert collection.count() == 0


def test_reset_collection_drops_previous_contents():
    """Regression: re-ingesting used to append a second copy of every document,
    doubling every count-style answer. A rebuild must start from empty.
    """
    client = chromadb.EphemeralClient()
    name = f"reingest-{uuid.uuid4().hex[:8]}"

    first = reset_collection(client, name)
    first.add(ids=["node-1"], embeddings=[[0.1, 0.2, 0.3]], documents=["2023 race one"])
    assert first.count() == 1

    second = reset_collection(client, name)
    assert second.count() == 0

    # A different node id for the same source document must not accumulate.
    second.add(ids=["node-2"], embeddings=[[0.1, 0.2, 0.3]], documents=["2023 race one"])
    assert second.count() == 1


def test_to_llama_documents_promotes_source_fields_into_metadata():
    raw = RawDocument(
        text="Monaco Grand Prix (2021).",
        source="ergast",
        source_id="2021-6-result",
        metadata={"season": "2021", "round": "6"},
    )
    (doc,) = to_llama_documents([raw])

    assert doc.doc_id == "ergast:2021-6-result"
    assert doc.metadata["source"] == "ergast"
    assert doc.metadata["source_id"] == "2021-6-result"
    assert doc.metadata["season"] == "2021"
