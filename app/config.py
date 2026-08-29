from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    redis_session_ttl: int = 3600

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str = "groovo-keypoints"

    # 환경별 Base URL (배포 환경에 따라 .env에서 주입)
    base_url_dev: str = "https://ai-dev.groovo.io"
    base_url_prod: str = "https://ai.groovo.io"

    # keypoint 비교 임계값
    feedback_threshold_good: float = 0.80
    feedback_threshold_bad: float = 0.60

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
