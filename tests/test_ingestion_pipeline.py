"""Offline tests for corpus assembly, with every source stubbed."""

import pytest

from f1_trivia_rag.ingestion import pipeline
from f1_trivia_rag.ingestion.common import RawDocument


def _doc(source, source_id):
    return RawDocument(text="t", source=source, source_id=source_id, metadata={})


@pytest.fixture
def stubbed_sources(monkeypatch):
    calls = {"reports": []}

    monkeypatch.setattr(
        pipeline,
        "fetch_season_results",
        lambda season, refresh=False: [_doc("ergast", f"{season}-1-result")],
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_season_schedule",
        lambda season, refresh=False: [
            {"raceName": "Monaco Grand Prix"},
            {"raceName": "Italian Grand Prix"},
        ],
    )

    def fake_reports(season, names):
        calls["reports"].append((season, names))
        return [_doc("wikipedia", f"{season} {name}") for name in names]

    monkeypatch.setattr(pipeline, "fetch_race_reports", fake_reports)
    return calls


def test_grand_prix_names_come_from_the_schedule(stubbed_sources):
    assert pipeline.grand_prix_names(2023) == ["Monaco Grand Prix", "Italian Grand Prix"]


def test_wikipedia_reports_are_actually_fetched(stubbed_sources):
    """Regression: --skip-wikipedia was a no-op because both branches skipped. Wikipedia
    ingestion has to genuinely happen when it is not skipped.
    """
    documents = pipeline.collect_documents([2023], include_wikipedia=True)

    sources = [doc.source for doc in documents]
    assert sources.count("wikipedia") == 2
    assert sources.count("ergast") == 1
    assert stubbed_sources["reports"] == [
        (2023, ["Monaco Grand Prix", "Italian Grand Prix"]),
    ]


def test_skip_wikipedia_skips_only_wikipedia(stubbed_sources):
    documents = pipeline.collect_documents([2023], include_wikipedia=False)

    assert [doc.source for doc in documents] == ["ergast"]
    assert stubbed_sources["reports"] == []


def test_multiple_seasons_are_each_fetched(stubbed_sources):
    documents = pipeline.collect_documents([2022, 2023], include_wikipedia=True)

    assert len(documents) == 6
    assert [season for season, _ in stubbed_sources["reports"]] == [2022, 2023]


def test_missing_reports_are_warned_about(monkeypatch, stubbed_sources, caplog):
    monkeypatch.setattr(pipeline, "fetch_race_reports", lambda season, names: [])

    with caplog.at_level("WARNING"):
        pipeline.collect_documents([2023], include_wikipedia=True)

    assert "0 of 2 race reports" in caplog.text
