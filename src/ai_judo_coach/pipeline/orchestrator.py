from collections.abc import Iterator

import numpy as np

from src.ai_judo_coach.config import(
    CLIP_STRIDE_SEC, 
    CLIP_DURATION_SEC,
    TARGET_FPS,
    DECORD_TARGET_DEVICE,
    YOLO_MODEL_WEIGHTS,
    YOLO_DEVICE,
    BYTETRACK_CONFIG_PATH,
    CLASSIFIER_DEVICE,
    JUDO_CLIPPER_MODEL_DIRECTORY
    
)
from src.ai_judo_coach.video import(
    compute_initial_clip_windows,
    cleanse_input_video,
    extract_frames_from_initial_window
)
from src.ai_judo_coach.schemas.internal import(
    InitialClipWindow,
    ClipProcessingResult,
    DetectedAttemptWindow,
    GeneratedAttemptClip
)
from src.ai_judo_coach.inference import(
    process_clip,
    resolve_yolo_device,
    load_yolo_model,
    construct_classifier
)


def run_pipeline(
    input_video_path: str,
    temporary_directory: str
) -> list[GeneratedAttemptClip]:
    """

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

    # instantiate models for inference
    yolo_device = resolve_yolo_device(
        requested_device=YOLO_DEVICE
    )
    yolo_model = load_yolo_model(
        yolo_model_path=YOLO_MODEL_WEIGHTS
    )

    # package already handles device
    judo_classifier_model = construct_classifier(
        classifier_release_directory=JUDO_CLIPPER_MODEL_DIRECTORY,
        classifier_device=CLASSIFIER_DEVICE
    )

    # should store the surviving InitialClipWindow objects and their probabilities (those that have a throw in them)
    initial_throw_attempt_intervals: list[DetectedAttemptWindow] = []

    for clip_interval in clip_windows_metadata_generator:

        # turn clip into bgr numpy list representation
        clip_interval_as_numpy: list[np.ndarray] = extract_frames_from_initial_window(
            source_video_path=cleansed_video_path,
            window=clip_interval,
            video_fps=float(TARGET_FPS),
            device=DECORD_TARGET_DEVICE
        )

        clip_result: ClipProcessingResult = process_clip(
            clip_as_numpy=clip_interval_as_numpy,
            clip_id=str(clip_interval.window_id),
            yolo_model=yolo_model,
            yolo_tracker_path=BYTETRACK_CONFIG_PATH,
            yolo_device=yolo_device,
            judo_clip_classifier=judo_classifier_model
        )


        if clip_result.contains_throw_attempt:
            initial_throw_attempt_intervals.append(
                DetectedAttemptWindow(
                    window=clip_interval,
                    attempt_probability=clip_result.attempt_probability
                )
            )



    
    
    