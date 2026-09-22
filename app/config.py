from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"

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

    # JSON 환경변수 예: {"hollywood-action":{"video_id":1,"keypoint_path":"refs/hollywood.npy"}}
    reference_keypoints: dict[str, dict[str, int | str]] = {}

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

if settings.environment == "production" and settings.jwt_secret == "change-me":
    raise RuntimeError(
        "JWT_SECRET이 기본값(change-me)입니다. 프로덕션에서는 .env에 Spring Boot와 "
        "동일한 실제 값을 설정해야 합니다."
    )
