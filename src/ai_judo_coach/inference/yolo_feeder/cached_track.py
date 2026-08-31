from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
    PersonDetection,
)

from .results_adapter import result_to_frame_detections


def collect_cached_tracked_clip_detections(
    yolo_model: YOLO,
    tracker_path: str,
    clip_as_numpy: list[np.ndarray],
    absolute_frame_indices: list[int],
    pose_detection_frame_indices: list[int],
    compute_device: str | int,
    pose_detection_cache: dict[int, FrameDetections],
    clip_id: str,
) -> ClipDetections:
    """
    Collect pose detections while reusing YOLO inference across
    overlapping clips.

    Only frames absent from the shared pose-detection cache are
    supplied as NumPy arrays. A fresh ByteTrack instance then replays
    all cached detections for the current clip, preserving per-window
    tracking semantics.
    """

    if len(clip_as_numpy) != len(
        pose_detection_frame_indices
    ):
        raise ValueError(
            "clip_as_numpy and pose_detection_frame_indices "
            "must have the same length"
        )

    _populate_pose_detection_cache(
        yolo_model=yolo_model,
        clip_as_numpy=clip_as_numpy,
        absolute_frame_indices=(
            pose_detection_frame_indices
        ),
        compute_device=compute_device,
        pose_detection_cache=pose_detection_cache,
    )

    missing_frame_indices = [
        frame_idx
        for frame_idx in absolute_frame_indices
        if frame_idx not in pose_detection_cache
    ]

    if missing_frame_indices:
        raise RuntimeError(
            "Pose detection cache is missing required frame "
            f"indices: {missing_frame_indices}"
        )

    tracker = _create_tracker(
        tracker_path=tracker_path,
    )

    tracked_frame_detections: list[FrameDetections] = []

    for relative_frame_idx, absolute_frame_idx in enumerate(
        absolute_frame_indices
    ):
        untracked_frame_detections = (
            pose_detection_cache[absolute_frame_idx]
        )

        tracked_frame_detections.append(
            _track_frame_detections(
                tracker=tracker,
                untracked_frame_detections=(
                    untracked_frame_detections
                ),
                relative_frame_idx=relative_frame_idx,
            )
        )

    return ClipDetections(
        frame_detections=tracked_frame_detections,
        clip_id=clip_id,
    )


def _populate_pose_detection_cache(
    yolo_model: YOLO,
    clip_as_numpy: list[np.ndarray],
    absolute_frame_indices: list[int],
    compute_device: str | int,
    pose_detection_cache: dict[int, FrameDetections],
) -> None:
    """
    Run pose inference for frames that are not already cached.

    Results are converted immediately into the project's lean CPU
    schemas so cached entries do not retain source images or GPU
    tensors.
    """

    for frame, absolute_frame_idx in zip(
        clip_as_numpy,
        absolute_frame_indices,
        strict=True,
    ):
        if absolute_frame_idx in pose_detection_cache:
            continue

        prediction_results = yolo_model.predict(
            source=frame,
            conf=0.1,
            batch=1,
            device=compute_device,
            verbose=False,
        )

        if len(prediction_results) != 1:
            raise RuntimeError(
                "Expected one YOLO result for one input frame, "
                f"received {len(prediction_results)}"
            )

        frame_detections = result_to_frame_detections(
            yolo_frame_result=prediction_results[0],
            frame_idx=absolute_frame_idx,
        )

        # The shared cache must never contain window-specific
        # ByteTrack IDs.
        for person_detection in (
            frame_detections.person_detections
        ):
            person_detection.track_id = None

        pose_detection_cache[absolute_frame_idx] = (
            frame_detections
        )


def _create_tracker(
    tracker_path: str,
) -> BYTETracker:
    """Construct a fresh ByteTrack instance from its YAML config."""

    tracker_config_path = Path(
        check_yaml(
            str(Path(tracker_path))
        )
    )

    tracker_arguments = IterableSimpleNamespace(
        **YAML.load(tracker_config_path)
    )

    if tracker_arguments.tracker_type != "bytetrack":
        raise ValueError(
            "Cached pose tracking requires a ByteTrack "
            f"configuration, got {tracker_arguments.tracker_type!r}"
        )

    return BYTETracker(
        args=tracker_arguments,
    )


def _track_frame_detections(
    tracker: BYTETracker,
    untracked_frame_detections: FrameDetections,
    relative_frame_idx: int,
) -> FrameDetections:
    """
    Replay one frame's cached detections through ByteTrack.

    The configured plain ByteTrack tracker does not require the
    original frame. This reproduces the relevant behaviour of
    Ultralytics' on_predict_postprocess_end tracking callback without
    retaining or decoding overlapping frames again.
    """

    tracker_input = _build_tracker_input(
        frame_detections=untracked_frame_detections,
    )

    tracks = tracker.update(
        tracker_input,
        None,
    )

    if len(tracks) == 0:
        return _copy_untracked_frame_detections(
            frame_detections=untracked_frame_detections,
            relative_frame_idx=relative_frame_idx,
        )

    return _build_tracked_frame_detections(
        tracks=tracks,
        untracked_frame_detections=(
            untracked_frame_detections
        ),
        relative_frame_idx=relative_frame_idx,
    )


def _build_tracker_input(
    frame_detections: FrameDetections,
) -> Boxes:
    """
    Convert project detections into the format expected by ByteTrack.

    Box columns are x1, y1, x2, y2, confidence and class.
    The pose model has only the person class, represented by class 0.
    """

    tracker_box_data = np.empty(
        (
            len(frame_detections.person_detections),
            6,
        ),
        dtype=np.float32,
    )

    for detection_idx, detection in enumerate(
        frame_detections.person_detections
    ):
        tracker_box_data[detection_idx, :4] = (
            detection.bbox_xyxy_px
        )
        tracker_box_data[detection_idx, 4] = (
            detection.bbox_conf
        )
        tracker_box_data[detection_idx, 5] = 0.0

    return Boxes(
        boxes=tracker_box_data,
        orig_shape=frame_detections.frame_shape_hw,
    )


def _copy_untracked_frame_detections(
    frame_detections: FrameDetections,
    relative_frame_idx: int,
) -> FrameDetections:
    """
    Copy original detections when ByteTrack returns no active tracks.

    Ultralytics retains the original untracked detections in this
    situation.
    """

    copied_person_detections = [
        PersonDetection(
            detection_idx=detection_idx,
            track_id=None,
            bbox_xyxy_px=(
                person_detection
                .bbox_xyxy_px
                .copy()
            ),
            bbox_xyxy_normalized=np.asarray(
                person_detection.bbox_xyxy_normalized,
                dtype=np.float32,
            ).copy(),
            bbox_conf=person_detection.bbox_conf,
            keypoints_xy_px=(
                person_detection
                .keypoints_xy_px
                .copy()
            ),
            keypoints_xy_norm=(
                person_detection
                .keypoints_xy_norm
                .copy()
            ),
            keypoints_conf=(
                person_detection
                .keypoints_conf
                .copy()
            ),
        )
        for detection_idx, person_detection in enumerate(
            frame_detections.person_detections
        )
    ]

    return FrameDetections(
        person_detections=copied_person_detections,
        frame_idx=relative_frame_idx,
        frame_shape_hw=frame_detections.frame_shape_hw,
    )


def _build_tracked_frame_detections(
    tracks: np.ndarray,
    untracked_frame_detections: FrameDetections,
    relative_frame_idx: int,
) -> FrameDetections:
    """
    Combine ByteTrack outputs with their corresponding cached poses.

    ByteTrack's final output column identifies the source detection
    whose keypoints belong to each tracked box.
    """

    frame_height, frame_width = (
        untracked_frame_detections.frame_shape_hw
    )

    tracked_person_detections: list[PersonDetection] = []

    for detection_idx, track in enumerate(tracks):
        if track.shape[0] < 8:
            raise ValueError(
                "Expected ByteTrack output with at least eight "
                f"columns, received shape {track.shape}"
            )

        source_detection_idx = int(track[-1])

        if not (
            0
            <= source_detection_idx
            < len(
                untracked_frame_detections
                .person_detections
            )
        ):
            raise ValueError(
                "ByteTrack returned an invalid source detection "
                f"index: {source_detection_idx}"
            )

        source_detection = (
            untracked_frame_detections
            .person_detections[source_detection_idx]
        )

        tracked_bbox_xyxy_px = np.asarray(
            track[:4],
            dtype=np.float32,
        ).copy()

        tracked_bbox_xyxy_px[[0, 2]] = np.clip(
            tracked_bbox_xyxy_px[[0, 2]],
            0.0,
            float(frame_width),
        )
        tracked_bbox_xyxy_px[[1, 3]] = np.clip(
            tracked_bbox_xyxy_px[[1, 3]],
            0.0,
            float(frame_height),
        )

        tracked_bbox_xyxy_normalized = (
            tracked_bbox_xyxy_px
            / np.asarray(
                [
                    frame_width,
                    frame_height,
                    frame_width,
                    frame_height,
                ],
                dtype=np.float32,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        tracked_person_detections.append(
            PersonDetection(
                detection_idx=detection_idx,
                track_id=int(track[4]),
                bbox_xyxy_px=tracked_bbox_xyxy_px,
                bbox_xyxy_normalized=(
                    tracked_bbox_xyxy_normalized
                ),
                bbox_conf=float(track[5]),
                keypoints_xy_px=(
                    source_detection
                    .keypoints_xy_px
                    .copy()
                ),
                keypoints_xy_norm=(
                    source_detection
                    .keypoints_xy_norm
                    .copy()
                ),
                keypoints_conf=(
                    source_detection
                    .keypoints_conf
                    .copy()
                ),
            )
        )

    return FrameDetections(
        person_detections=tracked_person_detections,
        frame_idx=relative_frame_idx,
        frame_shape_hw=(
            untracked_frame_detections.frame_shape_hw
        ),
    )
