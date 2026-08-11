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
        count = self._season_counts.get(where["season"], 0)
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
    assert metadata_filter.value == "2023"
    assert isinstance(metadata_filter.value, str), "Chroma matches metadata by type too"
    assert metadata_filter.operator.value == "=="


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
