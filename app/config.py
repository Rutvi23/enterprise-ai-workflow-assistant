from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "mock"  # mock | openai
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    retriever: str = "tfidf"  # tfidf | embeddings
    embedding_model: str = "text-embedding-3-small"
    min_score: float = 0.05
    top_k: int = 4

    chunk_size: int = 800
    chunk_overlap: int = 120
    max_upload_mb: int = 10

    max_agent_steps: int = 6
    fetch_allowlist: str = "api.github.com,httpbin.org"

    api_key: str = ""
    data_dir: str = "data"

    @property
    def allowed_hosts(self) -> set[str]:
        return {h.strip().lower() for h in self.fetch_allowlist.split(",") if h.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
