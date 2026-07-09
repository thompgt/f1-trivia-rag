from dataclasses import dataclass, field
from typing import Any


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
