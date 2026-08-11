"""Fetches race-report prose from Wikipedia to give the RAG index narrative context
that structured Ergast results don't have (incidents, controversies, weather, etc.).
"""

import wikipedia

from f1_trivia_rag.ingestion.common import RawDocument


def fetch_race_report(season: int, grand_prix: str) -> RawDocument | None:
    """`grand_prix` should match a Wikipedia title fragment, e.g. "Monaco Grand Prix"."""
    title = f"{season} {grand_prix}"
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return None

    return RawDocument(
        text=page.content,
        source="wikipedia",
        source_id=page.title,
        metadata={"season": str(season), "grand_prix": grand_prix, "url": page.url},
    )


def fetch_race_reports(season: int, grand_prix_names: list[str]) -> list[RawDocument]:
    documents = []
    for name in grand_prix_names:
        doc = fetch_race_report(season, name)
        if doc:
            documents.append(doc)
    return documents
