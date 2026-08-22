from typing import Any
from collections.abc import Iterator

import numpy as np
from ultralytics.engine.results import Results

from schemas import (
    PersonDetection,
    FrameDetections,
    ClipDetections,
)


def _to_numpy(value: Any) -> np.ndarray | None:
    """
    Convert torch/Ultralytics/numpy-like values to numpy.

    Returns None if value is None.
    """

    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)


def _get_frame_shape_hw(yolo_frame_result: Results) -> tuple[int, int]:
    """
    Extract frame shape from a YOLO Results object.

    Ultralytics stores this as (height, width).
    """

    return tuple(int(value) for value in yolo_frame_result.orig_shape)


def _require_not_none(value: Any, value_name: str, frame_idx: int) -> Any:
    """
    Raise a clear error if an expected YOLO field is missing.
    """

    if value is None:
        raise ValueError(
            f"Missing {value_name} in frame {frame_idx}"
        )

    return value


def _require_pose_keypoints(yolo_frame_result: Results, frame_idx: int) -> None:
    """
    Ensure that the YOLO Results object contains pose keypoints.

    If boxes exist but keypoints do not, something is wrong because this
    pipeline expects YOLO pose-model output.
    """

    if yolo_frame_result.keypoints is None:
        raise ValueError(
            f"Frame {frame_idx} has boxes but no keypoints. "
            "Expected YOLO pose model output."
        )


def _normalise_track_ids(
    track_ids: np.ndarray | None,
    n_people: int,
    frame_idx: int,
) -> list[int | None]:
    """
    Convert optional ByteTrack IDs into a list of length n_people.

    If ByteTrack did not assign IDs, returns a list of None values.
    """

    if track_ids is None:
        return [None] * n_people

    track_ids_list = list(track_ids.reshape(-1))

    if len(track_ids_list) != n_people:
        raise ValueError(
            f"Frame {frame_idx} has {n_people} detections but "
            f"{len(track_ids_list)} track IDs."
        )

    return track_ids_list


def _require_matching_detection_count(
    value: np.ndarray,
    value_name: str,
    expected_count: int,
    frame_idx: int,
) -> None:
    """
    Validate that a YOLO array has the expected number of detections
    along its first dimension.
    """

    actual_count = value.shape[0]

    if actual_count != expected_count:
        raise ValueError(
            f"Frame {frame_idx}: {value_name} has {actual_count} detections, "
            f"expected {expected_count}."
        )


def result_to_frame_detections(
    yolo_frame_result: Results,
    frame_idx: int,
) -> FrameDetections:
    """
    Takes in the yolo Results object for a given
    frame and returns a leaner FrameDetections object
    representing its values of interest
    """

    # unbox frame shape in (height, width) format as tuple
    frame_shape_hw = _get_frame_shape_hw(yolo_frame_result)

    # early guard in case no poses detected in frame
    # still want to return that there were none detected instead of skipping completely
    if yolo_frame_result.boxes is None or len(yolo_frame_result.boxes) == 0:
        return FrameDetections(
            person_detections=[],
            frame_idx=frame_idx,
            frame_shape_hw=frame_shape_hw,
        )

    _require_pose_keypoints(
        yolo_frame_result=yolo_frame_result,
        frame_idx=frame_idx,
    )

    person_detections_in_frame = []

    # unbox results of interest for persons in frame
    track_ids = _to_numpy(yolo_frame_result.boxes.id)

    bbx_confidence_scores = _require_not_none(
        value=_to_numpy(yolo_frame_result.boxes.conf),
        value_name="bbox confidence scores",
        frame_idx=frame_idx,
    )

    bbx_absolute_pixel_coords = _require_not_none(
        value=_to_numpy(yolo_frame_result.boxes.xyxy),
        value_name="bbox absolute pixel coordinates",
        frame_idx=frame_idx,
    )

    bbox_normalized_coords = _require_not_none(
        value=_to_numpy(yolo_frame_result.boxes.xyxyn),
        value_name="bbox normalized coordinates",
        frame_idx=frame_idx,
    )

    keypoints_raw = _require_not_none(
        value=_to_numpy(yolo_frame_result.keypoints.xy),
        value_name="raw keypoints",
        frame_idx=frame_idx,
    )

    keypoints_normalized = _require_not_none(
        value=_to_numpy(yolo_frame_result.keypoints.xyn),
        value_name="normalised keypoints",
        frame_idx=frame_idx,
    )

    keypoints_confidence_scores = _require_not_none(
        value=_to_numpy(yolo_frame_result.keypoints.conf),
        value_name="keypoint confidence scores",
        frame_idx=frame_idx,
    )

    n_people = bbx_absolute_pixel_coords.shape[0]

    _require_matching_detection_count(
        value=bbx_confidence_scores,
        value_name="bbox confidence scores",
        expected_count=n_people,
        frame_idx=frame_idx,
    )

    _require_matching_detection_count(
        value=bbox_normalized_coords,
        value_name="bbox normalized coordinates",
        expected_count=n_people,
        frame_idx=frame_idx,
    )

    _require_matching_detection_count(
        value=keypoints_raw,
        value_name="raw keypoints",
        expected_count=n_people,
        frame_idx=frame_idx,
    )

    _require_matching_detection_count(
        value=keypoints_normalized,
        value_name="normalised keypoints",
        expected_count=n_people,
        frame_idx=frame_idx,
    )

    _require_matching_detection_count(
        value=keypoints_confidence_scores,
        value_name="keypoint confidence scores",
        expected_count=n_people,
        frame_idx=frame_idx,
    )

    track_ids = _normalise_track_ids(
        track_ids=track_ids,
        n_people=n_people,
        frame_idx=frame_idx,
    )

    for person_idx in range(n_people):
        track_id = track_ids[person_idx]

        person_detected = PersonDetection(
            detection_idx=person_idx,
            track_id=track_id,
            bbox_xyxy_px=bbx_absolute_pixel_coords[person_idx],
            bbox_xyxy_normalized=bbox_normalized_coords[person_idx],
            bbox_conf=bbx_confidence_scores[person_idx],
            keypoints_xy_px=keypoints_raw[person_idx],
            keypoints_xy_norm=keypoints_normalized[person_idx],
            keypoints_conf=keypoints_confidence_scores[person_idx],
        )

        person_detections_in_frame.append(person_detected)

    frame_detections = FrameDetections(
        person_detections=person_detections_in_frame,
        frame_idx=frame_idx,
        frame_shape_hw=frame_shape_hw,
    )

    return frame_detections


def collect_clip_detections(
    clip_id: str,
    yolo_clip_output: Iterator[Results],
) -> ClipDetections:
    """
    Converts raw yolo output for clip into a
    leaner format containing all information of
    interest
    """

    frame_detections_entire_clip = []

    for frame_idx, results in enumerate(yolo_clip_output):
        frame_detections = result_to_frame_detections(
            yolo_frame_result=results,
            frame_idx=frame_idx,
        )

        frame_detections_entire_clip.append(frame_detections)

    clip_detections = ClipDetections(
        frame_detections=frame_detections_entire_clip,
        clip_id=clip_id,
    )

    return clip_detections
