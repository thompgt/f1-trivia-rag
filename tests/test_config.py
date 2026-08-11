from pathlib import Path

import pytest

from f1_trivia_rag.config import (
    PROJECT_ROOT,
    MissingApiKeyError,
    Settings,
    require_gemini_api_key,
    settings,
)


def test_settings_load():
    assert settings.chroma_collection == "f1_trivia"
    assert settings.chroma_persist_dir.name == "chroma"


def test_env_file_is_resolved_against_the_project_root_not_the_cwd():
    """Regression: a relative env_file meant running uvicorn or pytest from any other
    directory silently loaded no .env and left the API key empty.
    """
    env_file = Settings.model_config["env_file"]

    assert Path(env_file).is_absolute()
    assert Path(env_file) == PROJECT_ROOT / ".env"


def test_require_gemini_api_key_raises_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")

    with pytest.raises(MissingApiKeyError) as excinfo:
        require_gemini_api_key()

    # The error has to name the fix, not just the symptom.
    assert "GEMINI_API_KEY" in str(excinfo.value)
    assert ".env" in str(excinfo.value)


def test_require_gemini_api_key_returns_the_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    assert require_gemini_api_key() == "test-key"
