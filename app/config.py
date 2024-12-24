from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API Settings
    API_KEY_HEADER: str = "X-API-Key"
    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 1024 * 4  # 4GB

    # Processing Settings
    MAX_CONCURRENT_JOBS: int = 3
    CHUNK_SIZE: int = 1024 * 1024  # 1MB

    # Storage Settings
    UPLOAD_DIR: Path = Path("uploads")
    PROCESSED_DIR: Path = Path("processed")
    TEMP_DIR: Path = Path("temp")

    # Database
    DATABASE_URL: str = "postgresql://user:password@postgres/videodb"

    # Redis for job queue
    REDIS_URL: str = "redis://redis:6379/0"

    # Whisper Settings
    WHISPER_MODEL: str = "medium"

    # Grok API
    GROK_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()

# Create necessary directories
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.PROCESSED_DIR.mkdir(exist_ok=True)
settings.TEMP_DIR.mkdir(exist_ok=True)
