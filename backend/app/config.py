from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# backend/.env (repo root of backend)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    database_url: str = "sqlite:///./local.db"
    # Pydantic v2 is case-insensitive by default, so Database_URL from .env maps here.
    model_path: str = str(BASE_DIR / "model" / "amazon_tuned_hybrid_60_20_20.joblib")
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Gemini (Q&A). Reads Gemenai_Api_Key from backend/.env via validation_alias.
    gemini_api_key: str = Field(default="", validation_alias="Gemenai_Api_Key")
    gemini_model: str = "gemini-flash-lite-latest"
    # Comma-separated fallbacks tried when the primary model is unavailable
    # (e.g. 429 quota exhausted). Free-tier quota is per model, so falling
    # through to another model keeps the LLM fallback working.
    gemini_fallback_models: str = "gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest"
    gemini_timeout: float = 15.0

    @property
    def gemini_model_chain(self) -> list[str]:
        chain = [self.gemini_model]
        for name in (self.gemini_fallback_models or "").split(","):
            name = name.strip()
            if name and name not in chain:
                chain.append(name)
        return chain

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
