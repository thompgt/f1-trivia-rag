from f1_trivia_rag.config import settings


def test_settings_load():
    assert settings.chroma_collection == "f1_trivia"
    assert settings.chroma_persist_dir.name == "chroma"
