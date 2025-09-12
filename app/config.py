from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    database_url: str = "sqlite:///./golf_swing_analyzer.db"
    aws_access_key_id: str = "placeholder"
    aws_secret_access_key: str = "placeholder"
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "placeholder-bucket"
    secret_key: str = "your-secret-key-here"
    environment: str = "development"
    
    # Clerk configuration
    clerk_secret_key: str = "sk_test_placeholder"
    clerk_publishable_key: str = "pk_test_placeholder"
    
    class Config:
        env_file = ".env"


settings = Settings()
