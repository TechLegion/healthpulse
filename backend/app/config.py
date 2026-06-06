from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./vha.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501,http://127.0.0.1:8501"

    # Injected via Railway Variables (or local .env)
    openai_api_key: str | None = None
    groq_api_key: str | None = None

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.startswith("gsk_"))


settings = Settings()
