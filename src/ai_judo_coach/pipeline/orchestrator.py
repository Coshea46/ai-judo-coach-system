from collections.abc import Iterator

import numpy as np   # need for type hints only

from src.ai_judo_coach.config import(
    CLIP_STRIDE_SEC, 
    CLIP_DURATION_SEC,
    TARGET_FPS,
    DECORD_TARGET_DEVICE
)
from src.ai_judo_coach.video import(
    compute_initial_clip_windows,
    cleanse_input_video,
    extract_frames_from_initial_window
)
from src.ai_judo_coach.schemas.internal import InitialClipWindow


def run_pipeline(
    input_video_path: str,
    
):
    """
    WILL NEED TO UPDATE RETURN TYPE
    OF FUNCTION ONCE FULL PIPELINE BUILT
    """


    # cleanse input video first
    cleansed_video_path: str = cleanse_input_video(
        input_video_path=input_video_path
    )

    # compute initial clip windows on cleansed input video
    clip_windows_metadata_generator: Iterator[InitialClipWindow] = compute_initial_clip_windows(
        input_video_path=cleansed_video_path,
        individual_window_duration=float(CLIP_DURATION_SEC),
        stride=float(CLIP_STRIDE_SEC)
    )

    # should store the surviving InitialClipWindow objects (those that have a throw in them)
    initial_throw_attempt_intervals: list[InitialClipWindow] = []

    for clip_interval in clip_windows_metadata_generator:

        # turn clip into bgr numpy list representation
        clip_interval_as_numpy: list[np.ndarray] = extract_frames_from_initial_window(
            source_video_path=cleansed_video_path,
            window=clip_interval,
            video_fps=float(TARGET_FPS),
            device=DECORD_TARGET_DEVICE
        )

        # TODO: port over yolo and viterbi logic for single clip




    
    