from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_supervisor_model: str = "openai/gpt-oss-120b"
    groq_writer_model: str = "openai/gpt-oss-120b"

    tavily_api_key: str = ""

    database_url: str = "postgresql://tracker:tracker@localhost:5432/tracker"

    chroma_persist_dir: str = "./data/chroma"

    finbert_embed_model: str = "yiyanghkust/finbert-tone"
    finbert_sentiment_model: str = "ProsusAI/finbert"

    app_env: str = "dev"
    log_level: str = "INFO"


settings = Settings()
