"""Offline tests for SeasonAwareRetriever.

The retriever is the repo's core contribution and every test covering it used to be
live-API-gated. These use a stub index and a stub Chroma collection instead, so the
filter key/value/type and the top_k it asks for are pinned without embedding anything.
"""

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from f1_trivia_rag.rag.query_engine import (
    DEFAULT_TOP_K,
    MAX_SEASON_TOP_K,
    SeasonAwareRetriever,
    seasons_in_query,
)


class StubRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, query_bundle):
        return self._nodes


class StubIndex:
    """Records the kwargs `as_retriever` was called with."""

    def __init__(self, nodes_returned=0):
        self.calls = []
        self._nodes_returned = nodes_returned

    def as_retriever(self, **kwargs):
        self.calls.append(kwargs)
        nodes = [
            NodeWithScore(node=TextNode(text=f"node {i}"), score=1.0)
            for i in range(self._nodes_returned)
        ]
        return StubRetriever(nodes)


class StubCollection:
    """Minimal stand-in for a Chroma collection: counts ids per season."""

    def __init__(self, season_counts):
        self._season_counts = season_counts
        self.where_clauses = []

    def get(self, where, include):
        self.where_clauses.append(where)
        clause = where["season"]
        seasons = clause["$in"] if isinstance(clause, dict) else [clause]
        count = sum(self._season_counts.get(season, 0) for season in seasons)
        return {"ids": [str(i) for i in range(count)]}


def _retrieve(index, collection, question):
    retriever = SeasonAwareRetriever(index, collection)
    return retriever._retrieve(QueryBundle(question))


def test_no_season_in_query_uses_plain_similarity():
    index = StubIndex()
    _retrieve(index, StubCollection({}), "Who won at Monaco?")

    (call,) = index.calls
    assert call == {"similarity_top_k": DEFAULT_TOP_K}
    assert "filters" not in call


def test_season_in_query_applies_a_season_filter():
    index = StubIndex()
    _retrieve(index, StubCollection({"2023": 22}), "How many races did Red Bull win in 2023?")

    (call,) = index.calls
    (metadata_filter,) = call["filters"].filters
    assert metadata_filter.key == "season"
    assert metadata_filter.value == ["2023"]
    assert all(isinstance(v, str) for v in metadata_filter.value), "Chroma matches type too"
    assert metadata_filter.operator.value == "in"


def test_top_k_is_sized_from_the_stored_node_count():
    """Regression: top_k used to be a fixed 40 justified as '~24 rounds'. Nothing
    guarantees one node per race, so a season with more nodes than the cap was
    silently retrieved only in part.
    """
    index = StubIndex()
    _retrieve(index, StubCollection({"2023": 137}), "Who won the most races in 2023?")

    (call,) = index.calls
    assert call["similarity_top_k"] == 137


def test_top_k_never_drops_below_the_default():
    index = StubIndex()
    _retrieve(index, StubCollection({"2023": 1}), "Who won the 2023 opener?")

    assert index.calls[0]["similarity_top_k"] == DEFAULT_TOP_K


def test_top_k_is_capped_and_the_shortfall_is_logged(caplog):
    index = StubIndex()
    with caplog.at_level("WARNING"):
        _retrieve(index, StubCollection({"2023": MAX_SEASON_TOP_K + 10}), "2023 season summary?")

    assert index.calls[0]["similarity_top_k"] == MAX_SEASON_TOP_K
    assert "may undercount" in caplog.text


def test_partial_retrieval_is_logged(caplog):
    """If the store says 22 nodes exist for the season but retrieval returns 3, the
    aggregate answer is built on a subset - that must not pass silently.
    """
    index = StubIndex(nodes_returned=3)
    with caplog.at_level("WARNING"):
        nodes = _retrieve(index, StubCollection({"2023": 22}), "How many races in 2023?")

    assert len(nodes) == 3
    assert "retrieved 3 of 22" in caplog.text


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who won the 1950 British Grand Prix?", "1950"),
        ("Who won in 2099?", "2099"),
        ("Who won at Monaco?", None),
        ("Who finished P3 with 12 points?", None),
    ],
)
def test_season_detection(question, expected):
    index = StubIndex()
    collection = StubCollection({expected: 5} if expected else {})
    _retrieve(index, collection, question)

    if expected is None:
        assert collection.where_clauses == []
    else:
        assert collection.where_clauses == [{"season": expected}]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # A single season is still a single season.
        ("How many races did Red Bull win in 2023?", ["2023"]),
        # Regression: the first-match-wins regex answered only half of this.
        ("Compare 2021 and 2022", ["2021", "2022"]),
        ("Who won more races, Hamilton in 2018 or Verstappen in 2023?", ["2018", "2023"]),
        # Closed ranges expand.
        ("Who won the title between 2010 and 2013?", ["2010", "2011", "2012", "2013"]),
        ("Race wins from 2019 to 2021", ["2019", "2020", "2021"]),
        ("Champions 2005-2007", ["2005", "2006", "2007"]),
        # A bare "and" lists seasons, it does not span them.
        ("Who won in 2015 and 2023?", ["2015", "2023"]),
        # Open-ended scopes cannot be a set filter - better unfiltered than wrong.
        ("Who has the most wins since 2000?", []),
        ("Most poles before 1990?", []),
        ("Who has the most wins ever?", []),
        # No season at all.
        ("Who won at Monaco?", []),
    ],
)
def test_seasons_in_query(question, expected):
    assert seasons_in_query(question) == expected


def test_multi_season_query_filters_on_every_season():
    index = StubIndex()
    collection = StubCollection({"2021": 22, "2022": 22})
    _retrieve(index, collection, "Compare Red Bull in 2021 and 2022")

    (call,) = index.calls
    (metadata_filter,) = call["filters"].filters
    assert metadata_filter.value == ["2021", "2022"]
    assert metadata_filter.operator.value == "in"
    # top_k has to cover both seasons, not one of them.
    assert call["similarity_top_k"] == 44
    assert collection.where_clauses == [{"season": {"$in": ["2021", "2022"]}}]


def test_open_ended_range_does_not_narrow_to_the_named_season():
    """Regression: "most wins since 2000" filtered to the 2000 season alone."""
    index = StubIndex()
    collection = StubCollection({"2000": 17})
    _retrieve(index, collection, "Which driver has the most wins since 2000?")

    (call,) = index.calls
    assert "filters" not in call
    assert call["similarity_top_k"] == DEFAULT_TOP_K
    assert collection.where_clauses == []
