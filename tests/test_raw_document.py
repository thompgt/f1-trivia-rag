"""The `season` metadata invariant every source must satisfy.

Retrieval filters on `season`, and Chroma matches metadata by value *and* type, so a
single source storing ints instead of strings makes its documents invisible to every
season-scoped query - with no error anywhere. These tests pin the invariant.
"""

from f1_trivia_rag.ingestion.common import RawDocument


def _doc(season) -> RawDocument:
    return RawDocument(text="t", source="s", source_id="i", metadata={"season": season})


def test_int_season_is_normalised_to_str():
    assert _doc(2023).metadata["season"] == "2023"


def test_str_season_is_left_alone():
    assert _doc("2023").metadata["season"] == "2023"


def test_season_is_always_str_when_present():
    for season in (2023, "2023"):
        assert isinstance(_doc(season).metadata["season"], str)


def test_missing_season_is_not_invented():
    doc = RawDocument(text="t", source="s", source_id="i", metadata={"grand_prix": "Monaco"})
    assert "season" not in doc.metadata


def test_ergast_document_season_is_str():
    from f1_trivia_rag.ingestion.ergast import race_result_to_document

    doc = race_result_to_document(
        {
            "season": "2021",
            "round": "6",
            "raceName": "Monaco Grand Prix",
            "date": "2021-05-23",
            "Circuit": {"circuitName": "Circuit de Monaco"},
            "Results": [],
        }
    )
    assert isinstance(doc.metadata["season"], str)
