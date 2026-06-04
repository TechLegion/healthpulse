from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./vha.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501,http://127.0.0.1:8501"
    openai_api_key: str | None = None

settings = Settings()
