"""Pulls session-level facts (fastest laps, quali results) via FastF1.

FastF1 only has full timing data from ~2018 onward, and downloads are large, so this
is meant to supplement (not replace) the Ergast results for recent seasons.
"""


import fastf1

from f1_trivia_rag.config import settings
from f1_trivia_rag.ingestion.common import RawDocument

_CACHE_DIR = settings.data_raw_dir / ".fastf1_cache"


def _ensure_cache() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(_CACHE_DIR))


def fetch_session_summary(season: int, grand_prix: str, session_type: str = "R") -> RawDocument | None:
    """`session_type`: "R" (race), "Q" (qualifying), "SQ" (sprint), etc."""
    _ensure_cache()
    try:
        session = fastf1.get_session(season, grand_prix, session_type)
        session.load(telemetry=False, weather=False, messages=False)
    except Exception:
        return None

    results = session.results
    if results is None or results.empty:
        return None

    fastest = session.laps.pick_fastest() if not session.laps.empty else None

    lines = [f"{session.event['EventName']} ({season}) - {session.name} session summary."]
    for _, row in results.iterrows():
        lines.append(f"P{row['Position']:.0f}: {row['FullName']} ({row['TeamName']})")
    if fastest is not None:
        lines.append(
            f"Fastest lap: {fastest['Driver']} - {fastest['LapTime']}"
        )

    return RawDocument(
        text="\n".join(lines),
        source="fastf1",
        source_id=f"{season}-{grand_prix}-{session_type}",
        metadata={"season": season, "grand_prix": grand_prix, "session_type": session_type},
    )
