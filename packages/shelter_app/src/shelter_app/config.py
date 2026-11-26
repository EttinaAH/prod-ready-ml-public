"""Application settings, configurable through environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the shelter app.

    Every field can be overridden with an environment variable prefixed with
    `SHELTER_`, e.g. `SHELTER_MODEL_DIR=/tmp/models`.
    """

    model_config = SettingsConfigDict(env_prefix="SHELTER_")

    model_dir: Path = Path("models")
    default_model_name: str = "model"


def get_settings() -> Settings:
    """Return the current settings (used as a FastAPI dependency)."""
    return Settings()
