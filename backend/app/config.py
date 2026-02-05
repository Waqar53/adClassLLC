"""
Application Configuration

Centralized configuration management using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===========================================
    # APPLICATION
    # ===========================================
    APP_NAME: str = "AdClass AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # ===========================================
    # DATABASE
    # ===========================================
    DATABASE_URL: str = "postgresql+asyncpg://adclass:adclass_secret@localhost:5432/adclass_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False
    
    # ===========================================
    # REDIS
    # ===========================================
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour default
    
    # ===========================================
    # KAFKA
    # ===========================================
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "adclass-consumers"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    
    # ===========================================
    # META MARKETING API
    # ===========================================
    META_APP_ID: Optional[str] = None
    META_APP_SECRET: Optional[str] = None
    META_ACCESS_TOKEN: Optional[str] = None
    META_API_VERSION: str = "v18.0"
    
    # ===========================================
    # GOOGLE ADS API
    # ===========================================
    GOOGLE_ADS_DEVELOPER_TOKEN: Optional[str] = None
    GOOGLE_ADS_CLIENT_ID: Optional[str] = None
    GOOGLE_ADS_CLIENT_SECRET: Optional[str] = None
    GOOGLE_ADS_REFRESH_TOKEN: Optional[str] = None
    GOOGLE_ADS_LOGIN_CUSTOMER_ID: Optional[str] = None
    
    # ===========================================
    # TIKTOK ADS API
    # ===========================================
    TIKTOK_APP_ID: Optional[str] = None
    TIKTOK_APP_SECRET: Optional[str] = None
    TIKTOK_ACCESS_TOKEN: Optional[str] = None
    
    # ===========================================
    # ML MODELS
    # ===========================================
    MODEL_STORAGE_PATH: str = "/models"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    PREDICTIONS_CACHE_TTL: int = 1800  # 30 minutes
    
    # ===========================================
    # SECURITY
    # ===========================================
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ENCRYPTION_KEY: str = "your-encryption-key-32-bytes-long"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # ===========================================
    # STORAGE (S3/MinIO)
    # ===========================================
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "adclass-assets"
    S3_MODELS_BUCKET: str = "adclass-models"
    
    # ===========================================
    # OPTIMIZATION SETTINGS
    # ===========================================
    OPTIMIZATION_INTERVAL_MINUTES: int = 30
    BUDGET_CHANGE_THRESHOLD: float = 0.1  # 10% min change
    AUTO_PAUSE_ROAS_THRESHOLD: float = 0.5
    
    # ===========================================
    # CHURN DETECTION
    # ===========================================
    CHURN_CRITICAL_THRESHOLD: int = 30
    CHURN_WARNING_THRESHOLD: int = 50
    CHURN_MONITOR_THRESHOLD: int = 70
    
    # ===========================================
    # MONITORING
    # ===========================================
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL for Alembic migrations."""
        return self.DATABASE_URL.replace("+asyncpg", "")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid reading env vars on every request.
    """
    return Settings()


# Convenience instance
settings = get_settings()
