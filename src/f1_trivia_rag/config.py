from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Absolute, not ".env": a relative env_file is resolved against the working
    # directory, so running uvicorn or pytest from anywhere but the repo root loaded
    # no .env at all and left every setting on its default - including an empty API
    # key, which then failed deep inside a Gemini call (or, in the live tests, just
    # skipped the suite and looked green).
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_embed_model: str = "models/gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash"

    chroma_persist_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    chroma_collection: str = "f1_trivia"

    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"


settings = Settings()


class MissingApiKeyError(RuntimeError):
    """Raised when a Gemini call is about to be made without a configured key."""


def require_gemini_api_key() -> str:
    """Returns the Gemini API key, or fails loudly before anything tries to use it.

    An empty key is not a usable default: it produced an authentication failure from
    somewhere inside the embedding call, several frames from the actual cause. Checked
    at the point of configuration instead, so the error names the fix.
    """
    if not settings.gemini_api_key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env in "
            f"{PROJECT_ROOT} and fill in the key, or export GEMINI_API_KEY."
        )
    return settings.gemini_api_key
