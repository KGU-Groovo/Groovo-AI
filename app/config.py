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

    # 로컬 개발용 기준 안무. 배포에서는 S3 key를 REFERENCE_KEYPOINTS로 덮어쓴다.
    reference_keypoints: dict[str, dict[str, int | str]] = {
        "hollywood-action": {"video_id": 1, "keypoint_path": "data/references/hollywood-action.npy", "fps": 30},
        "rude": {"video_id": 2, "keypoint_path": "data/references/rude.npy", "fps": 30},
        "its-me": {"video_id": 3, "keypoint_path": "data/references/its-me.npy", "fps": 30},
        "wda": {"video_id": 4, "keypoint_path": "data/references/wda.npy", "fps": 30},
    }
    dca_checkpoint_path: str = "artifacts/training-normalized/checkpoints/best.pt"
    dca_device: str = "auto"
    dca_normalize_input: bool = False

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

if settings.environment == "production" and settings.jwt_secret == "change-me":
    raise RuntimeError(
        "JWT_SECRET이 기본값(change-me)입니다. 프로덕션에서는 .env에 Spring Boot와 "
        "동일한 실제 값을 설정해야 합니다."
    )
