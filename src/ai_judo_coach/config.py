# ----- Input video configs -----
TARGET_FPS: float = 30.0   # input videos should be forced into this frame rate before being processed

# ----- Initial video slice configs -----
CLIP_DURATION_SEC: float = 7.0
CLIP_STRIDE_SEC: float = 3.0  # slide window forward by this amount of seconds every time

# Decord configs
DECORD_TARGET_DEVICE: str = "gpu:0"  # try to use gpu for decord frame extraction
DECORD_FALLBACK_DEVICE: str = "cpu"  # default to cpu if gpu not available