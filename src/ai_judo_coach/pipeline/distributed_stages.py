"""Cloud-neutral stages used by distributed pipeline orchestration."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Final

from ai_judo_coach.attempt_clip_generation import (
    extract_final_clips,
    select_new_intervals,
)
from ai_judo_coach.config import (
    BYTETRACK_CONFIG_PATH,
    CLASSIFIER_DEVICE,
    CLIP_DURATION_SEC,
    CLIP_STRIDE_SEC,
    DECORD_TARGET_DEVICE,
    JUDO_CLIPPER_MODEL_DIRECTORY,
    MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC,
    MAX_GENERATED_ATTEMPT_CLIPS,
    OUTPUT_CLIP_NAMING_PATTERN,
    TARGET_FPS,
    YOLO_DEVICE,
    YOLO_MODEL_WEIGHTS,
)
from ai_judo_coach.inference import (
    construct_classifier,
    load_yolo_model,
    process_clip,
    resolve_yolo_device,
)
from ai_judo_coach.inference.yolo_feeder.cached_track import (
    collect_pose_detection_cache_from_frame_indices,
)
from ai_judo_coach.schemas.internal import (
    ClipProcessingResult,
    DetectedAttemptWindow,
    GeneratedAttemptClip,
    InitialClipWindow,
)
from ai_judo_coach.video import (
    compute_initial_clip_windows,
    compute_initial_window_frame_indices,
)


_FRAMES_PER_CLASSIFIER_WINDOW: Final[int] = 210
_POSE_INFERENCE_BATCH_SIZE: Final[int] = 8

type WindowWorkItem = tuple[
    InitialClipWindow,
    list[int],
]


def build_window_work_items(
    cleansed_video_path: str,
) -> list[WindowWorkItem]:
    """
    Build ordered classifier-window work items for a cleansed video.

    Every work item contains one seven-second window and exactly 210
    absolute source-frame indices at 30 FPS.
    """

    source_path = Path(cleansed_video_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Cleansed video does not exist: {source_path}"
        )

    clip_windows = list(
        compute_initial_clip_windows(
            input_video_path=str(source_path),
            individual_window_duration=float(
                CLIP_DURATION_SEC
            ),
            stride=float(CLIP_STRIDE_SEC),
        )
    )

    work_items: list[WindowWorkItem] = [
        (
            clip_window,
            compute_initial_window_frame_indices(
                window=clip_window,
                video_fps=float(TARGET_FPS),
            ),
        )
        for clip_window in clip_windows
    ]

    _validate_window_work_items(
        work_items=work_items,
        allow_empty=True,
    )

    return work_items


def process_window_group(
    cleansed_video_path: str,
    work_items: list[WindowWorkItem],
    stage_timings: dict[str, float] | None = None,
) -> list[ClipProcessingResult]:
    """
    Process one contiguous ordered group of classifier windows.

    Pose inference is performed once for the union of source frames
    required by this group. Cached untracked detections are then replayed
    through a fresh ByteTrack instance for every classifier window.

    This function is cloud-neutral. It does not access Modal, HTTP or S3.
    """

    worker_started_at = perf_counter()

    source_path = Path(cleansed_video_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Cleansed video does not exist: {source_path}"
        )

    _validate_window_work_items(
        work_items=work_items,
        allow_empty=False,
    )

    required_frame_indices = sorted(
        {
            frame_idx
            for _, absolute_frame_indices in work_items
            for frame_idx in absolute_frame_indices
        }
    )

    model_loading_started_at = perf_counter()

    yolo_device = resolve_yolo_device(
        requested_device=YOLO_DEVICE,
    )

    yolo_model = load_yolo_model(
        yolo_model_path=YOLO_MODEL_WEIGHTS,
    )

    judo_classifier_model = construct_classifier(
        classifier_release_directory=(
            JUDO_CLIPPER_MODEL_DIRECTORY
        ),
        classifier_device=CLASSIFIER_DEVICE,
    )

    model_loading_seconds = (
        perf_counter() - model_loading_started_at
    )

    pose_inference_started_at = perf_counter()

    pose_detection_cache = (
        collect_pose_detection_cache_from_frame_indices(
            yolo_model=yolo_model,
            source_video_path=str(source_path),
            required_frame_indices=required_frame_indices,
            compute_device=yolo_device,
            decoder_device=DECORD_TARGET_DEVICE,
            inference_batch_size=(
                _POSE_INFERENCE_BATCH_SIZE
            ),
        )
    )

    pose_inference_seconds = (
        perf_counter() - pose_inference_started_at
    )

    actual_cache_indices = sorted(
        pose_detection_cache
    )

    if actual_cache_indices != required_frame_indices:
        missing_frame_indices = sorted(
            set(required_frame_indices)
            - pose_detection_cache.keys()
        )
        unexpected_frame_indices = sorted(
            pose_detection_cache.keys()
            - set(required_frame_indices)
        )

        raise RuntimeError(
            "Pose detection cache does not exactly match the "
            "window group's required frame indices. "
            f"Missing: {missing_frame_indices}; "
            f"unexpected: {unexpected_frame_indices}"
        )

    window_processing_started_at = perf_counter()

    ordered_results: list[ClipProcessingResult] = []

    for clip_window, absolute_frame_indices in work_items:
        if len(absolute_frame_indices) != (
            _FRAMES_PER_CLASSIFIER_WINDOW
        ):
            raise RuntimeError(
                f"Window {clip_window.window_id} has "
                f"{len(absolute_frame_indices)} frames instead of "
                f"{_FRAMES_PER_CLASSIFIER_WINDOW}"
            )

        clip_result = process_clip(
            clip_as_numpy=[],
            clip_id=str(clip_window.window_id),
            yolo_model=yolo_model,
            yolo_tracker_path=BYTETRACK_CONFIG_PATH,
            yolo_device=yolo_device,
            judo_clip_classifier=judo_classifier_model,
            absolute_frame_indices=absolute_frame_indices,
            pose_detection_frame_indices=[],
            pose_detection_cache=pose_detection_cache,
        )

        if clip_result.clip_id != str(
            clip_window.window_id
        ):
            raise RuntimeError(
                "Window processing returned an unexpected clip ID: "
                f"{clip_result.clip_id!r} != "
                f"{str(clip_window.window_id)!r}"
            )

        ordered_results.append(
            clip_result
        )

    window_processing_seconds = (
        perf_counter() - window_processing_started_at
    )

    worker_total_seconds = (
        perf_counter() - worker_started_at
    )

    if stage_timings is not None:
        stage_timings.update(
            {
                "model_loading_seconds": (
                    model_loading_seconds
                ),
                "pose_inference_seconds": (
                    pose_inference_seconds
                ),
                "window_processing_seconds": (
                    window_processing_seconds
                ),
                "worker_total_seconds": (
                    worker_total_seconds
                ),
            }
        )

    return ordered_results


def finalise_window_results(
    cleansed_video_path: str,
    cleansed_video_duration: float,
    temporary_output_directory: str,
    work_items: list[WindowWorkItem],
    ordered_results: list[ClipProcessingResult],
) -> list[GeneratedAttemptClip]:
    """
    Select attempt intervals and extract final clips from ordered results.

    The supplied results must correspond one-to-one and in order with the
    supplied window work items.
    """

    source_path = Path(cleansed_video_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Cleansed video does not exist: {source_path}"
        )

    if (
        not math.isfinite(cleansed_video_duration)
        or cleansed_video_duration <= 0.0
    ):
        raise ValueError(
            "cleansed_video_duration must be a finite number "
            "greater than zero"
        )

    _validate_window_work_items(
        work_items=work_items,
        allow_empty=True,
    )

    if len(ordered_results) != len(work_items):
        raise RuntimeError(
            "Expected exactly one processing result per window: "
            f"{len(ordered_results)} results for "
            f"{len(work_items)} windows"
        )

    initial_throw_attempt_intervals: list[
        DetectedAttemptWindow
    ] = []

    for (
        clip_window,
        absolute_frame_indices,
    ), clip_result in zip(
        work_items,
        ordered_results,
        strict=True,
    ):
        expected_clip_id = str(
            clip_window.window_id
        )

        if clip_result.clip_id != expected_clip_id:
            raise RuntimeError(
                "Window result order or identity changed: "
                f"expected clip ID {expected_clip_id!r}, "
                f"received {clip_result.clip_id!r}"
            )

        if len(absolute_frame_indices) != (
            _FRAMES_PER_CLASSIFIER_WINDOW
        ):
            raise RuntimeError(
                f"Window {clip_window.window_id} has "
                f"{len(absolute_frame_indices)} frames instead of "
                f"{_FRAMES_PER_CLASSIFIER_WINDOW}"
            )

        if clip_result.contains_throw_attempt:
            initial_throw_attempt_intervals.append(
                DetectedAttemptWindow(
                    window=clip_window,
                    attempt_probability=(
                        clip_result.attempt_probability
                    ),
                )
            )

    if not initial_throw_attempt_intervals:
        return []

    selected_intervals = select_new_intervals(
        surviving_initial_windows=(
            initial_throw_attempt_intervals
        ),
        source_video_duration=(
            cleansed_video_duration
        ),
        max_new_interval_duration=(
            MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC
        ),
        max_intervals_per_video=(
            MAX_GENERATED_ATTEMPT_CLIPS
        ),
    )

    generated_clips_output_directory = (
        Path(temporary_output_directory)
        / "generated_clips"
    )

    return extract_final_clips(
        selected_intervals=selected_intervals,
        temporary_output_dir_path=str(
            generated_clips_output_directory
        ),
        clip_naming_pattern=(
            OUTPUT_CLIP_NAMING_PATTERN
        ),
        source_video_path=str(source_path),
        desired_fps=TARGET_FPS,
    )


def _validate_window_work_items(
    work_items: list[WindowWorkItem],
    allow_empty: bool,
) -> None:
    """
    Validate classifier-window identity, ordering and frame semantics.
    """

    if not work_items:
        if allow_empty:
            return

        raise ValueError(
            "work_items must contain at least one classifier window"
        )

    seen_window_ids: set[int] = set()
    previous_window: InitialClipWindow | None = None

    for clip_window, absolute_frame_indices in work_items:
        if clip_window.window_id in seen_window_ids:
            raise ValueError(
                "work_items contains duplicate window ID "
                f"{clip_window.window_id}"
            )

        seen_window_ids.add(
            clip_window.window_id
        )

        if len(absolute_frame_indices) != (
            _FRAMES_PER_CLASSIFIER_WINDOW
        ):
            raise ValueError(
                f"Window {clip_window.window_id} has "
                f"{len(absolute_frame_indices)} frames instead of "
                f"{_FRAMES_PER_CLASSIFIER_WINDOW}"
            )

        if not absolute_frame_indices:
            raise ValueError(
                f"Window {clip_window.window_id} has no frame indices"
            )

        if min(absolute_frame_indices) < 0:
            raise ValueError(
                f"Window {clip_window.window_id} contains a "
                "negative frame index"
            )

        if absolute_frame_indices != sorted(
            set(absolute_frame_indices)
        ):
            raise ValueError(
                f"Window {clip_window.window_id} frame indices "
                "must be unique and in ascending order"
            )

        expected_contiguous_indices = list(
            range(
                absolute_frame_indices[0],
                absolute_frame_indices[-1] + 1,
            )
        )

        if (
            absolute_frame_indices
            != expected_contiguous_indices
        ):
            raise ValueError(
                f"Window {clip_window.window_id} frame indices "
                "must be contiguous"
            )

        if previous_window is not None:
            if (
                clip_window.window_id
                != previous_window.window_id + 1
            ):
                raise ValueError(
                    "Window groups must contain consecutive ordered "
                    "window IDs"
                )

            if (
                clip_window.start_time
                < previous_window.start_time
            ):
                raise ValueError(
                    "Window groups must be ordered by start time"
                )

        previous_window = clip_window
