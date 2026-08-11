"""End-to-end tests: build a small real index and query it through the actual
Gemini-backed query engine and FastAPI app. Requires GEMINI_API_KEY (skipped otherwise)
since these exercise the live embedding/chat calls, not mocks.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from f1_trivia_rag.config import settings
from f1_trivia_rag.ingestion.common import RawDocument
from f1_trivia_rag.rag.build_index import build_index

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not settings.gemini_api_key,
        reason="GEMINI_API_KEY not set - skipping live end-to-end tests",
    ),
]

SAMPLE_DOCS = [
    RawDocument(
        text=(
            "Monaco Grand Prix (2021), held at Circuit de Monaco on 2021-05-23.\n"
            "P1: Max Verstappen (Red Bull) - Finished\n"
            "P2: Carlos Sainz (Ferrari) - Finished\n"
            "P3: Lando Norris (McLaren) - Finished"
        ),
        source="ergast",
        source_id="2021-6-result",
        metadata={"season": "2021", "round": "6", "race_name": "Monaco Grand Prix"},
    ),
    RawDocument(
        text=(
            "Azerbaijan Grand Prix (2021), held at Baku City Circuit on 2021-06-06.\n"
            "P1: Sergio Perez (Red Bull) - Finished\n"
            "P2: Sebastian Vettel (Aston Martin) - Finished\n"
            "P3: Pierre Gasly (AlphaTauri) - Finished"
        ),
        source="ergast",
        source_id="2021-8-result",
        metadata={"season": "2021", "round": "8", "race_name": "Azerbaijan Grand Prix"},
    ),
]


def _race_doc(season: str, round_: int, race_name: str, winner: str, winner_team: str) -> RawDocument:
    return RawDocument(
        text=(
            f"{race_name} ({season}), held at Some Circuit on {season}-01-0{round_}.\n"
            f"P1: {winner} ({winner_team}) - Finished\n"
            f"P2: Some Driver (Some Team) - Finished\n"
            f"P3: Another Driver (Another Team) - Finished"
        ),
        source="ergast",
        source_id=f"{season}-{round_}-result",
        metadata={"season": season, "round": str(round_), "race_name": race_name},
    )


# 9 races in a season Red Bull wins 7 of - more than the old fixed top_k=5, so a
# regression to plain similarity_top_k=5 would retrieve only a subset and could
# never report the true count of 7.
SEASON_DOCS = [
    _race_doc("2022", 1, "Race One", "Max Verstappen", "Red Bull"),
    _race_doc("2022", 2, "Race Two", "Max Verstappen", "Red Bull"),
    _race_doc("2022", 3, "Race Three", "Lewis Hamilton", "Mercedes"),
    _race_doc("2022", 4, "Race Four", "Max Verstappen", "Red Bull"),
    _race_doc("2022", 5, "Race Five", "Max Verstappen", "Red Bull"),
    _race_doc("2022", 6, "Race Six", "Charles Leclerc", "Ferrari"),
    _race_doc("2022", 7, "Race Seven", "Max Verstappen", "Red Bull"),
    _race_doc("2022", 8, "Race Eight", "Sergio Perez", "Red Bull"),
    _race_doc("2022", 9, "Race Nine", "Max Verstappen", "Red Bull"),
]


@pytest.fixture(scope="module")
def indexed_settings(tmp_path_factory):
    original_dir = settings.chroma_persist_dir
    original_collection = settings.chroma_collection

    settings.chroma_persist_dir = tmp_path_factory.mktemp("chroma_e2e")
    settings.chroma_collection = f"e2e-{uuid.uuid4().hex[:8]}"
    build_index(SAMPLE_DOCS + SEASON_DOCS)

    yield settings

    settings.chroma_persist_dir = original_dir
    settings.chroma_collection = original_collection


def test_query_engine_answers_known_fact(indexed_settings):
    from f1_trivia_rag.rag.query_engine import load_query_engine

    response = load_query_engine().query("Who won the 2021 Monaco Grand Prix?")
    assert "Verstappen" in str(response)
    assert len(response.source_nodes) > 0


def test_query_engine_cites_correct_source(indexed_settings):
    from f1_trivia_rag.rag.query_engine import load_query_engine

    response = load_query_engine().query("Who came second in the 2021 Azerbaijan Grand Prix?")
    sources = {node.metadata.get("source_id") for node in response.source_nodes}
    assert "2021-8-result" in sources


def test_aggregate_query_counts_full_season_not_just_top_k(indexed_settings):
    """Regression test: aggregate questions must reflect every race in the season,
    not just the `similarity_top_k` nearest chunks (previously hardcoded to 5).
    """
    from f1_trivia_rag.rag.query_engine import load_query_engine

    response = load_query_engine().query("How many races did Red Bull win in 2022?")
    assert "7" in str(response)
    season_2022_sources = {
        n.metadata.get("source_id")
        for n in response.source_nodes
        if n.metadata.get("source_id", "").startswith("2022-")
    }
    assert len(season_2022_sources) == len(SEASON_DOCS)


def test_chat_endpoint_end_to_end(indexed_settings, monkeypatch):
    import f1_trivia_rag.api.main as api_main

    monkeypatch.setattr(api_main, "_query_engine", None)
    client = TestClient(api_main.app)

    assert client.get("/health").status_code == 200

    resp = client.post("/chat", json={"message": "Who won the 2021 Monaco Grand Prix?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "Verstappen" in body["answer"]
    assert len(body["citations"]) > 0
