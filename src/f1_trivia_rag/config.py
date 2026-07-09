from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_embed_model: str = "models/gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash"

    chroma_persist_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    chroma_collection: str = "f1_trivia"

    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"


settings = Settings()
