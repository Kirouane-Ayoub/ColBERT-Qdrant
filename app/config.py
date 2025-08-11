import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    # Server config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "info"

    # ColBERT model config
    MODEL_NAME: str = "colbert-ir/colbertv2.0"
    MAX_BATCH_SIZE: int = 32
    MAX_QUERY_LENGTH: int = 512
    MAX_DOCUMENT_LENGTH: int = 512

    # Qdrant config
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    DEFAULT_COLLECTION: str = "colbert_embeddings"
    VECTOR_SIZE: int = 128

    # Health check
    HEALTH_CHECK_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
