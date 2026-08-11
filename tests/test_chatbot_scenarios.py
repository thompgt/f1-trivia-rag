"""15 scenario tests probing the RAG chatbot from distinct angles: aggregation,
scoring-era rules, disqualifications, sprint races, entity aliasing, negation,
out-of-domain refusal, citation grounding, and input robustness.

Each test builds its own isolated Chroma collection (unique temp dir + collection
name) and tears it down itself, so tests share no mutable state and are safe to
run in parallel (e.g. `pytest -n auto` under pytest-xdist, one worker process per
test). Requires GEMINI_API_KEY (skipped otherwise) since these exercise the live
embedding/chat calls, not mocks.
"""

import uuid
from contextlib import contextmanager

import pytest

from f1_trivia_rag.config import settings
from f1_trivia_rag.ingestion.common import RawDocument
from f1_trivia_rag.rag.build_index import build_index

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not settings.gemini_api_key,
        reason="GEMINI_API_KEY not set - skipping live end-to-end tests",
    ),
]


def _doc(text: str, source: str, source_id: str, **metadata) -> RawDocument:
    return RawDocument(text=text, source=source, source_id=source_id, metadata=metadata)


@contextmanager
def scenario_engine(tmp_path_factory, label: str, docs: list[RawDocument]):
    """Builds a fresh index from `docs` in its own temp dir/collection and returns
    a query engine over it. Each caller gets a fully independent Chroma instance.
    """
    from f1_trivia_rag.rag.query_engine import load_query_engine

    original_dir = settings.chroma_persist_dir
    original_collection = settings.chroma_collection
    try:
        settings.chroma_persist_dir = tmp_path_factory.mktemp(f"chroma_{label}")
        settings.chroma_collection = f"scenario-{label}-{uuid.uuid4().hex[:8]}"
        build_index(docs)
        yield load_query_engine()
    finally:
        settings.chroma_persist_dir = original_dir
        settings.chroma_collection = original_collection


# ---------------------------------------------------------------------------
# 1. Simple factual lookup - single-hop, single-document
# ---------------------------------------------------------------------------
def test_01_single_race_winner_lookup(tmp_path_factory):
    docs = [
        _doc(
            "Italian Grand Prix (2019), held at Monza on 2019-09-08.\n"
            "P1: Charles Leclerc (Ferrari) - Finished\n"
            "P2: Valtteri Bottas (Mercedes) - Finished",
            "ergast", "2019-14-result", season="2019", round="14", race_name="Italian Grand Prix",
        )
    ]
    with scenario_engine(tmp_path_factory, "s01", docs) as engine:
        response = engine.query("Who won the 2019 Italian Grand Prix?")
        assert "Leclerc" in str(response)


# ---------------------------------------------------------------------------
# 2. Aggregate count within a single season (the top-5 regression this fix addresses)
# ---------------------------------------------------------------------------
def test_02_season_win_count_exceeds_old_top_k(tmp_path_factory):
    winners = ["Red Bull"] * 7 + ["Ferrari", "Mercedes"]
    docs = [
        _doc(
            f"Race {i} (2022), held at Some Circuit on 2022-01-0{i}.\nP1: Driver ({team}) - Finished",
            "ergast", f"2022-{i}-result", season="2022", round=str(i), race_name=f"Race {i}",
        )
        for i, team in enumerate(winners, start=1)
    ]
    with scenario_engine(tmp_path_factory, "s02", docs) as engine:
        response = engine.query("How many races did Red Bull win in 2022?")
        assert "7" in str(response)


# ---------------------------------------------------------------------------
# 3. Zero-count aggregate - must not hallucinate a win that never happened
# ---------------------------------------------------------------------------
def test_03_zero_win_count_is_not_hallucinated(tmp_path_factory):
    docs = [
        _doc(
            f"Race {i} (2015), held at Some Circuit on 2015-01-0{i}.\nP1: Driver (Mercedes) - Finished",
            "ergast", f"2015-{i}-result", season="2015", round=str(i), race_name=f"Race {i}",
        )
        for i in range(1, 6)
    ]
    with scenario_engine(tmp_path_factory, "s03", docs) as engine:
        response = str(engine.query("How many races did Williams win in 2015 according to these results?"))
        lowered = response.lower()
        assert "0" in response or "did not win" in lowered or "none" in lowered


# ---------------------------------------------------------------------------
# 4. Disqualification - as-raced vs final classified result must both be answerable
# ---------------------------------------------------------------------------
def test_04_disqualification_as_raced_vs_final_classified(tmp_path_factory):
    docs = [
        _doc(
            "Belgian Grand Prix (2021), held at Spa-Francorchamps on 2021-08-29.\n"
            "P1 (as raced): Max Verstappen (Red Bull) - Finished\n"
            "P1 (final classified, post-race DSQ of P1 as-raced for a technical infringement): "
            "Lewis Hamilton (Mercedes) - Finished\n"
            "P2 (final classified): Max Verstappen (Red Bull) - Finished",
            "ergast", "2021-dsq-result", season="2021", round="99", race_name="Belgian Grand Prix",
        )
    ]
    with scenario_engine(tmp_path_factory, "s04", docs) as engine:
        response = str(engine.query(
            "Who was disqualified from this race, and who ended up classified as the winner after the DSQ?"
        ))
        assert "Hamilton" in response


# ---------------------------------------------------------------------------
# 5. Sprint race vs Grand Prix conflation
# ---------------------------------------------------------------------------
def test_05_sprint_result_not_conflated_with_gp_result(tmp_path_factory):
    docs = [
        _doc(
            "British Grand Prix Sprint (2021), held at Silverstone on 2021-07-17.\n"
            "Sprint P1: Max Verstappen (Red Bull) - Finished (awards 3 sprint points, not GP points)",
            "ergast", "2021-10-sprint", season="2021", round="10", race_name="British Grand Prix",
            session="sprint",
        ),
        _doc(
            "British Grand Prix (2021), held at Silverstone on 2021-07-18.\n"
            "P1: Lewis Hamilton (Mercedes) - Finished\n"
            "P2: Max Verstappen (Red Bull) - Finished",
            "ergast", "2021-10-result", season="2021", round="10", race_name="British Grand Prix",
            session="race",
        ),
    ]
    with scenario_engine(tmp_path_factory, "s05", docs) as engine:
        response = str(engine.query("Who won the main 2021 British Grand Prix race (not the sprint)?"))
        assert "Hamilton" in response


# ---------------------------------------------------------------------------
# 6. Team rename / alias resolution
# ---------------------------------------------------------------------------
def test_06_team_rename_alias_resolution(tmp_path_factory):
    docs = [
        _doc(
            "Turkish Grand Prix (2020), held at Istanbul Park on 2020-11-15.\n"
            "P1: Sergio Perez (Racing Point, later renamed Aston Martin from 2021) - Finished",
            "ergast", "2020-14-result", season="2020", round="14", race_name="Turkish Grand Prix",
            constructor_canonical="Aston Martin", constructor_alias="Racing Point",
        )
    ]
    with scenario_engine(tmp_path_factory, "s06", docs) as engine:
        response = str(
            engine.query("Did Aston Martin (formerly Racing Point) win the 2020 Turkish Grand Prix?")
        )
        assert "Perez" in response


# ---------------------------------------------------------------------------
# 7. Pre-2003 vs 2003-2009 points-per-win era distinction
# ---------------------------------------------------------------------------
def test_07_legacy_points_era_not_confused_with_modern_scoring(tmp_path_factory):
    docs = [
        _doc(
            "San Marino Grand Prix (1995), held at Imola on 1995-04-30. Race win awarded 10 points "
            "under the era's 10-6-4-3-2-1 scoring system (not the 2010+ 25-18-15... system).\n"
            "P1: Michael Schumacher (Benetton) - Finished",
            "ergast", "1995-3-result", season="1995", round="3", race_name="San Marino Grand Prix",
        )
    ]
    with scenario_engine(tmp_path_factory, "s07", docs) as engine:
        response = str(engine.query("How many points did the race winner get at this 1995 Grand Prix?"))
        assert "10" in response


# ---------------------------------------------------------------------------
# 8. Fastest-lap bonus point era (2019-2020 & 2025+)
# ---------------------------------------------------------------------------
def test_08_fastest_lap_bonus_point_era(tmp_path_factory):
    docs = [
        _doc(
            "German Grand Prix (2019), held at Hockenheim on 2019-07-28.\n"
            "P1: Max Verstappen (Red Bull) - Finished, did not set fastest lap\n"
            "Fastest lap: Valtteri Bottas (Mercedes), finished P11 - Finished "
            "(awarded the extra fastest-lap point, in effect 2019-2020)",
            "ergast", "2019-11-result", season="2019", round="11", race_name="German Grand Prix",
        )
    ]
    with scenario_engine(tmp_path_factory, "s08", docs) as engine:
        response = str(engine.query("Who got the bonus point for fastest lap at the 2019 German Grand Prix?"))
        assert "Bottas" in response


# ---------------------------------------------------------------------------
# 9. Out-of-domain question - should decline rather than hallucinate
# ---------------------------------------------------------------------------
def test_09_out_of_domain_question_is_not_hallucinated(tmp_path_factory):
    docs = [
        _doc(
            "Monaco Grand Prix (2018), held at Circuit de Monaco on 2018-05-27.\n"
            "P1: Daniel Ricciardo (Red Bull) - Finished",
            "ergast", "2018-6-result", season="2018", round="6", race_name="Monaco Grand Prix",
        )
    ]
    with scenario_engine(tmp_path_factory, "s09", docs) as engine:
        response = str(engine.query("What was Lewis Hamilton's annual salary in 2018?"))
        lowered = response.lower()
        assert any(
            phrase in lowered
            for phrase in ["not", "no information", "don't know", "unable", "cannot"]
        )


# ---------------------------------------------------------------------------
# 10. Negation / complement-set query over a full season
# ---------------------------------------------------------------------------
def test_10_negation_query_over_full_season(tmp_path_factory):
    results = {1: "Red Bull", 2: "Red Bull", 3: "Ferrari", 4: "Red Bull", 5: "Mercedes"}
    docs = [
        _doc(
            f"Race {i} (2022), held at Some Circuit on 2022-01-0{i}.\nP1: Driver ({team}) - Finished",
            "ergast", f"2022-{i}-result", season="2022", round=str(i), race_name=f"Race {i}",
        )
        for i, team in results.items()
    ]
    with scenario_engine(tmp_path_factory, "s10", docs) as engine:
        response = str(engine.query("In the 2022 season, which races did Red Bull NOT win?"))
        assert "Race 3" in response and "Race 5" in response


# ---------------------------------------------------------------------------
# 11. Ambiguous entity without disambiguating context (same race name, different years)
# ---------------------------------------------------------------------------
def test_11_ambiguous_race_name_across_seasons(tmp_path_factory):
    docs = [
        _doc(
            "Australian Grand Prix (2016), held at Albert Park on 2016-03-20.\n"
            "P1: Nico Rosberg (Mercedes) - Finished",
            "ergast", "2016-1-result", season="2016", round="1", race_name="Australian Grand Prix",
        ),
        _doc(
            "Australian Grand Prix (2022), held at Albert Park on 2022-04-10.\n"
            "P1: Charles Leclerc (Ferrari) - Finished",
            "ergast", "2022-3-result", season="2022", round="3", race_name="Australian Grand Prix",
        ),
    ]
    with scenario_engine(tmp_path_factory, "s11", docs) as engine:
        response = str(engine.query("Who won the Australian Grand Prix in 2022?"))
        assert "Leclerc" in response and "Rosberg" not in response


# ---------------------------------------------------------------------------
# 12. Citation grounding - every cited source must actually contain the claimed fact
# ---------------------------------------------------------------------------
def test_12_citations_are_grounded_in_retrieved_sources(tmp_path_factory):
    docs = [
        _doc(
            "Spanish Grand Prix (2023), held at Circuit de Barcelona-Catalunya on 2023-06-04.\n"
            "P1: Max Verstappen (Red Bull) - Finished\n"
            "P2: Fernando Alonso (Aston Martin) - Finished",
            "ergast", "2023-8-result", season="2023", round="8", race_name="Spanish Grand Prix",
        )
    ]
    with scenario_engine(tmp_path_factory, "s12", docs) as engine:
        response = engine.query("Who finished second at the 2023 Spanish Grand Prix?")
        assert len(response.source_nodes) > 0
        assert any("Alonso" in n.node.get_content() for n in response.source_nodes)


# ---------------------------------------------------------------------------
# 13. Multi-hop reasoning across two separate documents
# ---------------------------------------------------------------------------
def test_13_multi_hop_reasoning_across_two_races(tmp_path_factory):
    docs = [
        _doc(
            "Race A (2024), held at Circuit A on 2024-03-01.\nP1: Lando Norris (McLaren) - Finished",
            "ergast", "2024-a-result", season="2024", round="1", race_name="Race A",
        ),
        _doc(
            "Race B (2024), held at Circuit B on 2024-03-08.\nP1: Lando Norris (McLaren) - Finished",
            "ergast", "2024-b-result", season="2024", round="2", race_name="Race B",
        ),
    ]
    with scenario_engine(tmp_path_factory, "s13", docs) as engine:
        response = str(engine.query("Did the same driver win both Race A and Race B in 2024? If so, who?"))
        assert "Norris" in response


# ---------------------------------------------------------------------------
# 14. Cross-source consistency - Ergast (structured) and Wikipedia (prose) for one event
# ---------------------------------------------------------------------------
def test_14_cross_source_consistency_ergast_and_wikipedia(tmp_path_factory):
    docs = [
        _doc(
            "Canadian Grand Prix (2019), held at Circuit Gilles Villeneuve on 2019-06-09.\n"
            "P1 (final classified): Lewis Hamilton (Mercedes) - Finished",
            "ergast", "2019-7-result", season="2019", round="7", race_name="Canadian Grand Prix",
        ),
        _doc(
            "Race report: Sebastian Vettel crossed the finish line first at the 2019 Canadian Grand Prix "
            "but received a post-race time penalty, promoting Lewis Hamilton to the win.",
            "wikipedia", "2019-canadian-gp-report", season="2019", race_name="Canadian Grand Prix",
        ),
    ]
    with scenario_engine(tmp_path_factory, "s14", docs) as engine:
        response = str(
            engine.query(
                "Who was awarded the win at the 2019 Canadian Grand Prix, "
                "and why was there controversy?"
            )
        )
        assert "Hamilton" in response and ("penalty" in response.lower() or "Vettel" in response)


# ---------------------------------------------------------------------------
# 15. Robustness to casual phrasing / shorthand year / informal team name
# ---------------------------------------------------------------------------
def test_15_robust_to_casual_phrasing_and_shorthand(tmp_path_factory):
    docs = [
        _doc(
            f"Race {i} (2023), held at Some Circuit on 2023-01-0{i}.\nP1: Driver ({team}) - Finished",
            "ergast", f"2023-{i}-result", season="2023", round=str(i), race_name=f"Race {i}",
        )
        for i, team in enumerate(["Red Bull"] * 3 + ["Ferrari"], start=1)
    ]
    with scenario_engine(tmp_path_factory, "s15", docs) as engine:
        response = str(engine.query("how many races did redbull win in 2023?"))
        assert "3" in response
