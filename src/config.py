"""
Configuration for Groovo DCA-Net v2 Prototype.

이 파일은 DCA-Net v2 프로젝트의 기준 설정을 고정한다.
실제 모델, 데이터셋, 학습 코드는 다음 단계에서 작성한다.
"""

from pathlib import Path


# ============================================================
# Project Identity
# ============================================================

PROJECT_NAME = "Groovo DCA-Net v2 Prototype"
MODEL_VERSION = "dca_v2_33"
MODEL_NAME = "Groovo DCA-Net v2 Prototype"


# ============================================================
# Input Specification
# ============================================================

FRAME_LENGTH = 30
NUM_JOINTS = 33
COORD_DIM = 3
INPUT_SHAPE = (FRAME_LENGTH, NUM_JOINTS, COORD_DIM)


# ============================================================
# Model Architecture Settings
# ============================================================

EMBED_DIM = 128
NUM_HEADS = 4
SPATIAL_LAYERS = 2
TEMPORAL_LAYERS = 2
DROPOUT = 0.1

USE_SPATIAL_ENCODER = True
USE_TEMPORAL_ENCODER = True
USE_CROSS_ATTENTION = True


# ============================================================
# Training Settings
# ============================================================

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# Normalization Settings
# ============================================================

CENTER_MODE = "hip_center"
SCALE_MODE = "shoulder_width"

# MediaPipe Pose index
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

EPS = 1e-8


# ============================================================
# Output Settings
# ============================================================

SCORE_MIN = 0.0
SCORE_MAX = 100.0

GREEN_THRESHOLD = 80.0
ORANGE_THRESHOLD = 60.0

TOP_K_HIGHLIGHT = 3


# ============================================================
# Path Settings
# ============================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/Groovo/groovo_dca_v2")

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REFERENCE_DIR = DATA_DIR / "reference_library"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"
EVAL_DIR = OUTPUT_DIR / "evaluation"
INFERENCE_DIR = OUTPUT_DIR / "inference"

DOCS_DIR = PROJECT_ROOT / "docs"
EXPORT_DIR = PROJECT_ROOT / "server_export"


# ============================================================
# Utility Functions
# ============================================================

def get_color_from_score(score_100):
    """
    0~100 점수를 green / orange / red로 변환한다.
    color는 모델이 직접 분류하지 않고 score 기준 후처리로 만든다.
    """
    score_100 = float(score_100)

    if score_100 >= GREEN_THRESHOLD:
        return "green"
    if score_100 >= ORANGE_THRESHOLD:
        return "orange"
    return "red"


def get_project_paths():
    """
    프로젝트 주요 경로를 dict로 반환한다.
    """
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "SRC_DIR": SRC_DIR,
        "DATA_DIR": DATA_DIR,
        "RAW_DIR": RAW_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "SYNTHETIC_DIR": SYNTHETIC_DIR,
        "REFERENCE_DIR": REFERENCE_DIR,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "FIGURE_DIR": FIGURE_DIR,
        "EVAL_DIR": EVAL_DIR,
        "INFERENCE_DIR": INFERENCE_DIR,
        "DOCS_DIR": DOCS_DIR,
        "EXPORT_DIR": EXPORT_DIR,
    }


def ensure_project_dirs():
    """
    프로젝트 주요 폴더를 생성한다.
    Colab 런타임이 초기화되어도 안전하게 다시 실행할 수 있다.
    """
    for path in get_project_paths().values():
        path.mkdir(parents=True, exist_ok=True)


def print_config_summary():
    """
    DCA-Net v2 설정 요약을 출력한다.
    """
    print("=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)
    print(f"Model name      : {MODEL_NAME}")
    print(f"Model version   : {MODEL_VERSION}")
    print(f"Input shape     : {INPUT_SHAPE}")
    print()
    print("[Input]")
    print(f"- user_seq       : {INPUT_SHAPE}")
    print(f"- idol_seq       : {INPUT_SHAPE}")
    print(f"- joints         : MediaPipe Pose {NUM_JOINTS} joints")
    print(f"- coordinate dim : {COORD_DIM}")
    print()
    print("[Normalization]")
    print(f"- center mode    : {CENTER_MODE}")
    print(f"- scale mode     : {SCALE_MODE}")
    print(f"- hip indices    : left={LEFT_HIP_INDEX}, right={RIGHT_HIP_INDEX}")
    print(f"- shoulder idx   : left={LEFT_SHOULDER_INDEX}, right={RIGHT_SHOULDER_INDEX}")
    print()
    print("[Target Architecture]")
    print(f"- Spatial Encoder              : {USE_SPATIAL_ENCODER}")
    print(f"- Temporal Encoder             : {USE_TEMPORAL_ENCODER}")
    print(f"- Choreography Cross-Attention : {USE_CROSS_ATTENTION}")
    print("- Feedback Decoder             : planned")
    print("- Output Head                  : score + highlight targets")
    print()
    print("[Model Hyperparameters]")
    print(f"- embed dim      : {EMBED_DIM}")
    print(f"- heads          : {NUM_HEADS}")
    print(f"- spatial layers : {SPATIAL_LAYERS}")
    print(f"- temporal layers: {TEMPORAL_LAYERS}")
    print(f"- dropout        : {DROPOUT}")
    print()
    print("[Output]")
    print("- score_100        : model output")
    print("- color            : post-processing from score_100")
    print("- highlight_joints : top-k high error joints")
    print("- highlight_coords : frontend visualization coordinates")
    print("- joint_errors     : per-joint error values")
    print()
    print("[Color Rule]")
    print(f"- score >= {GREEN_THRESHOLD:.0f}       : green")
    print(f"- {ORANGE_THRESHOLD:.0f} <= score < {GREEN_THRESHOLD:.0f} : orange")
    print(f"- score < {ORANGE_THRESHOLD:.0f}        : red")
    print("=" * 70)


# 비율 검증
assert abs((TRAIN_RATIO + VALID_RATIO + TEST_RATIO) - 1.0) < 1e-6, \
    "TRAIN_RATIO + VALID_RATIO + TEST_RATIO must be 1.0"

assert NUM_JOINTS == 33, "DCA v2는 MediaPipe 33개 관절 기준이어야 함"
assert FRAME_LENGTH == 30, "DCA v2 기본 입력 프레임 길이는 30이어야 함"
assert COORD_DIM == 3, "DCA v2 기본 좌표 차원은 3이어야 함"
