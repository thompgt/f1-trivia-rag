from dataclasses import dataclass, field
from typing import Any

SEASON_KEY = "season"


@dataclass
class RawDocument:
    """Normalized ingestion output, independent of any vector-store/index library.

    `source` and `source_id` let build_index.py cite exactly where a fact came from
    (e.g. source="ergast", source_id="2021-monaco-race-result").
    """

    text: str
    source: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # `season` is the one metadata key retrieval filters on, and Chroma compares
        # metadata by value *and* type: a filter of season == "2023" does not match a
        # node stored with season == 2023. The sources disagreed - Ergast yields JSON
        # strings while the Wikipedia and FastF1 helpers passed their int argument
        # straight through - which made those documents silently invisible to every
        # season-scoped query. Normalising here, at the single contract every source
        # funnels through, makes the invariant impossible for a new source to break.
        season = self.metadata.get(SEASON_KEY)
        if season is not None:
            self.metadata[SEASON_KEY] = str(season)
