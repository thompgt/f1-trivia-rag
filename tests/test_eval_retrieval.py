"""Offline tests for the retrieval-eval metrics and dataset."""

from pathlib import Path

import pytest

from f1_trivia_rag.eval.retrieval import (
    EvalQuestion,
    QuestionResult,
    evaluate,
    load_questions,
)

QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "evals" / "retrieval_questions.jsonl"


def _result(retrieved, expected):
    return QuestionResult(
        question=EvalQuestion(id="q", question="?"),
        retrieved_source_ids=retrieved,
        expected_source_ids=expected,
    )


def test_hit_requires_one_relevant_document():
    assert _result(["a", "b"], ["b"]).hit
    assert not _result(["a", "b"], ["c"]).hit


def test_recall_is_the_share_of_relevant_documents_found():
    assert _result(["a", "b"], ["a", "b", "c", "d"]).recall == 0.5
    assert _result(["a", "b", "c"], ["a", "b", "c"]).recall == 1.0
    assert _result([], ["a"]).recall == 0.0


def test_a_partial_season_hits_but_does_not_recall():
    """The distinction the whole harness exists for: retrieving 5 of 22 races still
    'hits', and still produces a wrong count.
    """
    result = _result([f"2023-{i}-result" for i in range(1, 6)], [f"2023-{i}-result" for i in range(1, 23)])

    assert result.hit
    assert result.recall < 0.25


def test_reciprocal_rank_rewards_ranking_the_answer_first():
    assert _result(["a", "b", "c"], ["a"]).reciprocal_rank == 1.0
    assert _result(["a", "b", "c"], ["c"]).reciprocal_rank == pytest.approx(1 / 3)
    assert _result(["a"], ["z"]).reciprocal_rank == 0.0


def test_missing_source_ids_are_reported():
    assert _result(["a"], ["a", "b", "c"]).missing_source_ids == ["b", "c"]


def test_evaluate_truncates_at_k():
    questions = [EvalQuestion(id="q", question="?", relevant_source_ids=("d",))]

    report = evaluate(questions, lambda _q: ["a", "b", "c", "d"], k=3)

    assert report.hit_rate == 0.0
    assert report.results[0].retrieved_source_ids == ["a", "b", "c"]


def test_evaluate_resolves_full_season_labels_against_the_corpus():
    questions = [EvalQuestion(id="agg", question="How many in 2023?", full_season="2023")]
    stored = ["2023-1-result", "2023-2-result", "2023-3-result"]

    report = evaluate(
        questions,
        lambda _q: stored[:2],
        k=10,
        season_source_ids=lambda _season: stored,
    )

    assert report.recall == pytest.approx(2 / 3)
    assert report.failures[0].missing_source_ids == ["2023-3-result"]


def test_full_season_label_without_a_resolver_is_an_error():
    questions = [EvalQuestion(id="agg", question="?", full_season="2023")]

    with pytest.raises(ValueError, match="full_season"):
        evaluate(questions, lambda _q: [], k=5)


def test_report_aggregates_across_questions():
    questions = [
        EvalQuestion(id="a", question="?", relevant_source_ids=("x",)),
        EvalQuestion(id="b", question="?", relevant_source_ids=("y",)),
    ]

    report = evaluate(questions, lambda _q: ["x"], k=5)

    assert report.hit_rate == 0.5
    assert report.recall == 0.5
    assert "hit rate@5" in report.format()
    assert "b" in report.format(), "failing question ids must be listed"


def test_the_committed_question_set_loads_and_is_labelled():
    questions = load_questions(QUESTIONS_FILE)

    assert len(questions) >= 10
    assert len({q.id for q in questions}) == len(questions), "ids must be unique"
    for question in questions:
        assert question.question.strip()
        assert question.relevant_source_ids or question.full_season, (
            f"{question.id} has no labels, so it scores nothing"
        )


def test_the_question_set_covers_both_lookups_and_aggregates():
    questions = load_questions(QUESTIONS_FILE)

    assert any(q.full_season for q in questions)
    assert any(q.relevant_source_ids and not q.full_season for q in questions)
    assert any(len(q.relevant_source_ids) > 1 for q in questions), "multi-document cases"
