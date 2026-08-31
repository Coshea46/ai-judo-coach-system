from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ai_judo_coach.config import(
    CLIP_STRIDE_SEC,
    CLIP_DURATION_SEC,
    TARGET_FPS,
    DECORD_TARGET_DEVICE,
    YOLO_MODEL_WEIGHTS,
    YOLO_DEVICE,
    BYTETRACK_CONFIG_PATH,
    CLASSIFIER_DEVICE,
    JUDO_CLIPPER_MODEL_DIRECTORY,
    MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC,
    MAX_GENERATED_ATTEMPT_CLIPS,
    OUTPUT_CLIP_NAMING_PATTERN
)
from ai_judo_coach.video import(
    compute_initial_clip_windows,
    cleanse_input_video,
    extract_frames_from_initial_window
)
from ai_judo_coach.schemas.internal import(
    InitialClipWindow,
    ClipProcessingResult,
    DetectedAttemptWindow,
    GeneratedAttemptClip
)
from ai_judo_coach.inference import(
    process_clip,
    resolve_yolo_device,
    load_yolo_model,
    construct_classifier
)
from ai_judo_coach.inference.inference_schemas import (
    FrameDetections,
)
from ai_judo_coach.attempt_clip_generation import(
    select_new_intervals,
    extract_final_clips
)


def run_pipeline(
    input_video_path: str,
    temporary_output_directory: str
) -> list[GeneratedAttemptClip]:
    """
    Process one input video and generate clips containing predicted
    throw attempts.

    The input video is cleansed, divided into overlapping initial
    windows, and processed through pose estimation, player detection,
    and clip classification. Positively classified windows are then
    consolidated into final intervals and extracted as .mp4 files in
    the supplied temporary output directory.

    Returns:
        A list of generated attempt clips. Returns an empty list when
        no throw attempts are detected.
    """

    # cleanse input video first
    cleansed_video_path: str
    cleansed_video_path, cleansed_video_duration = cleanse_input_video(
        input_video_path=input_video_path,
        output_directory=temporary_output_directory
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

    # Cache untracked pose detections by absolute frame index so
    # overlapping windows do not repeat YOLO pose inference.
    pose_detection_cache: dict[int, FrameDetections] = {}

    for clip_interval in clip_windows_metadata_generator:

        # turn clip into bgr numpy list representation
        clip_interval_as_numpy: list[np.ndarray] = extract_frames_from_initial_window(
            source_video_path=cleansed_video_path,
            window=clip_interval,
            video_fps=float(TARGET_FPS),
            device=DECORD_TARGET_DEVICE
        )

        start_frame_index = int(
            clip_interval.start_time * float(TARGET_FPS)
        )

        absolute_frame_indices = list(
            range(
                start_frame_index,
                start_frame_index
                + len(clip_interval_as_numpy),
            )
        )

        clip_result: ClipProcessingResult = process_clip(
            clip_as_numpy=clip_interval_as_numpy,
            clip_id=str(clip_interval.window_id),
            yolo_model=yolo_model,
            yolo_tracker_path=BYTETRACK_CONFIG_PATH,
            yolo_device=yolo_device,
            judo_clip_classifier=judo_classifier_model,
            absolute_frame_indices=absolute_frame_indices,
            pose_detection_cache=pose_detection_cache
        )

        if clip_result.contains_throw_attempt:
            initial_throw_attempt_intervals.append(
                DetectedAttemptWindow(
                    window=clip_interval,
                    attempt_probability=clip_result.attempt_probability
                )
            )

    # no detected attempts is a valid pipeline result
    if not initial_throw_attempt_intervals:
        return []

    # now create clips to return to the user
    revised_intervals = select_new_intervals(
        surviving_initial_windows=initial_throw_attempt_intervals,
        source_video_duration=cleansed_video_duration,
        max_new_interval_duration=MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC,
        max_intervals_per_video=MAX_GENERATED_ATTEMPT_CLIPS
    )

    generated_clips_output_directory = str(
        Path(temporary_output_directory)
        / "generated_clips"
    )

    final_clips = extract_final_clips(
        selected_intervals=revised_intervals,
        temporary_output_dir_path=generated_clips_output_directory,
        clip_naming_pattern=OUTPUT_CLIP_NAMING_PATTERN,
        source_video_path=cleansed_video_path,
        desired_fps=TARGET_FPS
    )

    return final_clips
