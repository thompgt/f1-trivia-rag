from f1_trivia_rag.ingestion.ergast import race_result_to_document

SAMPLE_RACE = {
    "season": "2021",
    "round": "6",
    "raceName": "Monaco Grand Prix",
    "date": "2021-05-23",
    "Circuit": {"circuitName": "Circuit de Monaco"},
    "Results": [
        {
            "position": "1",
            "status": "Finished",
            "Driver": {"givenName": "Max", "familyName": "Verstappen"},
            "Constructor": {"name": "Red Bull"},
        }
    ],
}


def test_race_result_to_document():
    doc = race_result_to_document(SAMPLE_RACE)
    assert doc.source == "ergast"
    assert doc.source_id == "2021-6-result"
    assert "Verstappen" in doc.text
    assert doc.metadata["season"] == "2021"
