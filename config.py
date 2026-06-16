from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://memory:memory@localhost:5432/memory_engine"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_port: int = 50051

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Extraction LLM
    anthropic_api_key: str = ""
    extraction_model: str = "claude-haiku-4-5"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Memory engine
    max_token_budget: int = 1500
    token_budget_hard_ceiling: int = 2000
    procedural_min_successes: int = 3
    dedup_similarity_threshold: float = 0.92
    embedding_sync_threshold: float = 0.8

    # Auth
    secret_key: str = "change_this_in_production"


settings = Settings()
