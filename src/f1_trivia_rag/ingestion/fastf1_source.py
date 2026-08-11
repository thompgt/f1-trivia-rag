"""Pulls session-level facts (fastest laps, quali results) via FastF1.

FastF1 only has full timing data from ~2018 onward, and downloads are large, so this
is meant to supplement (not replace) the Ergast results for recent seasons.
"""

import logging

import fastf1
import requests
from fastf1.req import RateLimitExceededError

from f1_trivia_rag.config import settings
from f1_trivia_rag.ingestion.common import RawDocument

logger = logging.getLogger(__name__)

_CACHE_DIR = settings.data_raw_dir / ".fastf1_cache"

# What "this session isn't available" actually looks like coming out of FastF1:
# ValueError for an event or session that does not exist, KeyError for a schedule row
# missing a field, and the network/rate-limit failures underneath. Anything else is a
# bug in this code or in FastF1, and should not be flattened into "no data".
_UNAVAILABLE = (
    ValueError,
    KeyError,
    RateLimitExceededError,
    requests.RequestException,
)


def _ensure_cache() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(_CACHE_DIR))


def fetch_session_summary(
    season: int, grand_prix: str, session_type: str = "R"
) -> RawDocument | None:
    """`session_type`: "R" (race), "Q" (qualifying), "SQ" (sprint), etc.

    Returns None when the session genuinely has no data. A bare `except Exception`
    used to flatten every failure - including a typo in this module - into that same
    None, so a broken source was indistinguishable from an empty one.
    """
    _ensure_cache()
    try:
        session = fastf1.get_session(season, grand_prix, session_type)
        session.load(telemetry=False, weather=False, messages=False)
    except _UNAVAILABLE:
        logger.warning(
            "FastF1 has no %s session for %s %s.",
            session_type,
            season,
            grand_prix,
            exc_info=True,
        )
        return None

    results = session.results
    if results is None or results.empty:
        logger.info("FastF1 returned no results for %s %s %s.", season, grand_prix, session_type)
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
        metadata={"season": str(season), "grand_prix": grand_prix, "session_type": session_type},
    )
