"""Offline retrieval evaluation: does the retriever fetch the documents that contain
the answer, before any LLM is involved?

Retrieval is the part of this system that silently degrades. A chunking change, a new
`MAX_SEASON_TOP_K`, or a different embedding model can all make answers worse while
every test still passes, because the tests assert on generated text and the model is
good at sounding right over a partial context. These metrics put a number on it:

- **hit rate@k** - the share of questions where at least one relevant document was
  retrieved. The floor: below this, the answer cannot be grounded at all.
- **recall@k** - the share of each question's relevant documents that were retrieved,
  averaged. This is the one that matters for aggregates: "how many races did Red Bull
  win in 2023" needs *all* of them, not one.
- **MRR** - how high the first relevant document ranked, which is what a reranker
  would move.

The labels are `source_id`s, so they survive re-ingestion and re-chunking - node ids do
not.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    """One labelled question.

    `relevant_source_ids` are the documents that must be retrieved to answer it.
    `full_season` marks aggregate questions, where the whole season is the answer's
    evidence and partial retrieval means a wrong count rather than a vaguer answer.
    """

    id: str
    question: str
    relevant_source_ids: tuple[str, ...] = ()
    full_season: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> EvalQuestion:
        return cls(
            id=row["id"],
            question=row["question"],
            relevant_source_ids=tuple(row.get("relevant_source_ids", ())),
            full_season=row.get("full_season"),
            note=row.get("note", ""),
        )


@dataclass
class QuestionResult:
    question: EvalQuestion
    retrieved_source_ids: list[str]
    expected_source_ids: list[str]

    @property
    def hit(self) -> bool:
        return bool(set(self.retrieved_source_ids) & set(self.expected_source_ids))

    @property
    def recall(self) -> float:
        if not self.expected_source_ids:
            return 1.0
        found = set(self.retrieved_source_ids) & set(self.expected_source_ids)
        return len(found) / len(set(self.expected_source_ids))

    @property
    def reciprocal_rank(self) -> float:
        expected = set(self.expected_source_ids)
        for rank, source_id in enumerate(self.retrieved_source_ids, start=1):
            if source_id in expected:
                return 1.0 / rank
        return 0.0

    @property
    def missing_source_ids(self) -> list[str]:
        retrieved = set(self.retrieved_source_ids)
        return [s for s in self.expected_source_ids if s not in retrieved]


@dataclass
class EvalReport:
    k: int
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return _mean([1.0 if r.hit else 0.0 for r in self.results])

    @property
    def recall(self) -> float:
        return _mean([r.recall for r in self.results])

    @property
    def mrr(self) -> float:
        return _mean([r.reciprocal_rank for r in self.results])

    @property
    def failures(self) -> list[QuestionResult]:
        return [r for r in self.results if r.recall < 1.0]

    def format(self) -> str:
        lines = [
            f"questions      {len(self.results)}",
            f"hit rate@{self.k:<4}  {self.hit_rate:.3f}",
            f"recall@{self.k:<6}  {self.recall:.3f}",
            f"MRR            {self.mrr:.3f}",
        ]
        if self.failures:
            lines.append("")
            lines.append("incomplete retrievals:")
            for result in self.failures:
                missing = ", ".join(result.missing_source_ids[:5])
                more = "" if len(result.missing_source_ids) <= 5 else " ..."
                lines.append(
                    f"  {result.question.id:<24} recall {result.recall:.2f}  missing: {missing}{more}"
                )
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_questions(path: Path) -> list[EvalQuestion]:
    """Reads the labelled set from JSONL, ignoring blank lines and `#` comments."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        questions.append(EvalQuestion.from_dict(json.loads(stripped)))
    return questions


def evaluate(
    questions: Iterable[EvalQuestion],
    retrieve: Callable[[str], list[str]],
    *,
    k: int,
    season_source_ids: Callable[[str], list[str]] | None = None,
) -> EvalReport:
    """Runs `retrieve` (question -> retrieved source_ids, best first) over `questions`.

    `season_source_ids` resolves a `full_season` label to every source_id stored for
    that season, so aggregate questions are scored against the real corpus rather than
    a list that goes stale the moment a season is re-ingested.
    """
    report = EvalReport(k=k)

    for question in questions:
        expected = list(question.relevant_source_ids)
        if question.full_season:
            if season_source_ids is None:
                raise ValueError(
                    f"{question.id} is labelled full_season but no resolver was provided"
                )
            expected = sorted(set(expected) | set(season_source_ids(question.full_season)))

        retrieved = retrieve(question.question)[:k]
        report.results.append(
            QuestionResult(
                question=question,
                retrieved_source_ids=retrieved,
                expected_source_ids=expected,
            )
        )

    return report
