"""System configurations"""

from pathlib import Path


# ----- Input video configs -----
TARGET_FPS: float = 30.0   # input videos should be forced into this frame rate before being processed

# ----- Initial video slice configs -----
CLIP_DURATION_SEC: float = 7.0
CLIP_STRIDE_SEC: float = 3.0  # slide window forward by this amount of seconds every time

# Decord configs
DECORD_TARGET_DEVICE: str = "gpu:0"  # try to use gpu for decord frame extraction
DECORD_FALLBACK_DEVICE: str = "cpu"  # default to cpu if gpu not available

# JudoClipClassifier configs
CLASSIFIER_DEVICE: str = 'auto'
YOLO_DEVICE: str = ''
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WEIGHTS_DIRECTORY = _PROJECT_ROOT / "weights"

JUDO_CLIPPER_MODEL_DIRECTORY: str = str(
    _WEIGHTS_DIRECTORY
    / "judo_clipper_classification_model_v1"
)

YOLO_MODEL_WEIGHTS: str = str(
    _WEIGHTS_DIRECTORY
    / "ultralytics_v11x_yolo"
    / "yolo11x-pose.pt"
)

BYTETRACK_CONFIG_PATH: str = str(
    _PROJECT_ROOT
    / "src"
    / "ai_judo_coach"
    / "inference"
    / "configs"
    / "trackers"
    / "bytetrack.yaml"
)


# final clip generation configs
MAX_GENERATED_ATTEMPT_CLIPS: int = 10
MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC: float = 14.0
