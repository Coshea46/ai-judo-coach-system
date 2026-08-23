from types import SimpleNamespace
from unittest.mock import call

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
    PersonDetection,
)
from ai_judo_coach.inference.yolo_feeder.results_adapter import (
    _get_frame_shape_hw,
    _to_numpy,
    collect_clip_detections,
    result_to_frame_detections,
)


RESULTS_ADAPTER_MODULE_PATH = (
    "ai_judo_coach.inference.yolo_feeder."
    "results_adapter"
)


class _FakeBoxes:
    """Minimal substitute for the fields used from YOLO Boxes."""

    def __init__(
        self,
        detection_count: int,
        track_ids: object,
        confidence_scores: object,
        pixel_coordinates: object,
        normalized_coordinates: object,
    ) -> None:
        self._detection_count = detection_count
        self.id = track_ids
        self.conf = confidence_scores
        self.xyxy = pixel_coordinates
        self.xyxyn = normalized_coordinates

    def __len__(self) -> int:
        return self._detection_count


class _FakeTensor:
    """Tensor-like value used to verify conversion to NumPy."""

    def __init__(
        self,
        value: np.ndarray,
    ) -> None:
        self.value = value
        self.detach_called = False
        self.cpu_called = False
        self.numpy_called = False

    def detach(self) -> "_FakeTensor":
        self.detach_called = True
        return self

    def cpu(self) -> "_FakeTensor":
        self.cpu_called = True
        return self

    def numpy(self) -> np.ndarray:
        self.numpy_called = True
        return self.value


def _create_yolo_result(
    detection_count: int = 2,
    track_ids: object = ...,
) -> SimpleNamespace:
    """Create a minimal YOLO-like result for adapter tests."""

    bbox_confidence = np.array(
        [0.91, 0.82],
        dtype=np.float32,
    )[:detection_count]

    bbox_xyxy_px = np.array(
        [
            [10.0, 20.0, 110.0, 220.0],
            [200.0, 100.0, 350.0, 400.0],
        ],
        dtype=np.float32,
    )[:detection_count]

    bbox_xyxy_normalized = np.array(
        [
            [0.01, 0.02, 0.11, 0.22],
            [0.20, 0.10, 0.35, 0.40],
        ],
        dtype=np.float32,
    )[:detection_count]

    keypoints_xy_px = np.zeros(
        (detection_count, 17, 2),
        dtype=np.float32,
    )
    keypoints_xy_norm = np.zeros(
        (detection_count, 17, 2),
        dtype=np.float32,
    )
    keypoints_conf = np.zeros(
        (detection_count, 17),
        dtype=np.float32,
    )

    for person_idx in range(detection_count):
        for keypoint_idx in range(17):
            keypoints_xy_px[
                person_idx,
                keypoint_idx,
            ] = [
                (
                    10.0
                    + person_idx
                    + keypoint_idx
                ),
                (
                    20.0
                    + person_idx
                    + keypoint_idx
                ),
            ]
            keypoints_xy_norm[
                person_idx,
                keypoint_idx,
            ] = [
                (
                    0.10
                    + (person_idx * 0.10)
                    + (keypoint_idx * 0.001)
                ),
                (
                    0.20
                    + (person_idx * 0.10)
                    + (keypoint_idx * 0.001)
                ),
            ]
            keypoints_conf[
                person_idx,
                keypoint_idx,
            ] = (
                0.50
                + (person_idx * 0.10)
                + (keypoint_idx * 0.01)
            )

    if track_ids is ...:
        track_ids = np.array(
            [11.0, 22.0],
            dtype=np.float32,
        )[:detection_count]

    return SimpleNamespace(
        orig_shape=(720, 1280),
        boxes=_FakeBoxes(
            detection_count=detection_count,
            track_ids=track_ids,
            confidence_scores=bbox_confidence,
            pixel_coordinates=bbox_xyxy_px,
            normalized_coordinates=(
                bbox_xyxy_normalized
            ),
        ),
        keypoints=SimpleNamespace(
            xy=keypoints_xy_px,
            xyn=keypoints_xy_norm,
            conf=keypoints_conf,
        ),
    )


def test_to_numpy_returns_none_for_none() -> None:
    assert _to_numpy(None) is None


def test_to_numpy_returns_numpy_array_unchanged() -> None:
    expected = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    result = _to_numpy(expected)

    assert result is expected


def test_to_numpy_detaches_moves_to_cpu_and_converts_tensor() -> None:
    expected = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )
    tensor_like_value = _FakeTensor(
        value=expected,
    )

    result = _to_numpy(
        tensor_like_value
    )

    assert result is expected
    assert tensor_like_value.detach_called
    assert tensor_like_value.cpu_called
    assert tensor_like_value.numpy_called


def test_to_numpy_converts_array_like_value() -> None:
    result = _to_numpy(
        [1.0, 2.0, 3.0]
    )

    assert isinstance(result, np.ndarray)

    np.testing.assert_array_equal(
        result,
        np.array(
            [1.0, 2.0, 3.0],
        ),
    )


def test_get_frame_shape_returns_integer_height_and_width() -> None:
    yolo_result = SimpleNamespace(
        orig_shape=(
            np.int64(720),
            np.int64(1280),
        ),
    )

    result = _get_frame_shape_hw(
        yolo_frame_result=yolo_result,
    )

    assert result == (720, 1280)
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)


def test_result_to_frame_detections_converts_all_people() -> None:
    yolo_result = _create_yolo_result(
        detection_count=2,
    )

    result = result_to_frame_detections(
        yolo_frame_result=yolo_result,
        frame_idx=4,
    )

    assert isinstance(
        result,
        FrameDetections,
    )
    assert result.frame_idx == 4
    assert result.frame_shape_hw == (
        720,
        1280,
    )
    assert len(result.person_detections) == 2

    first_detection = (
        result.person_detections[0]
    )
    second_detection = (
        result.person_detections[1]
    )

    assert isinstance(
        first_detection,
        PersonDetection,
    )
    assert first_detection.detection_idx == 0
    assert first_detection.track_id == 11
    assert first_detection.bbox_conf == pytest.approx(
        0.91
    )

    np.testing.assert_array_equal(
        first_detection.bbox_xyxy_px,
        yolo_result.boxes.xyxy[0],
    )
    np.testing.assert_array_equal(
        first_detection.bbox_xyxy_normalized,
        yolo_result.boxes.xyxyn[0],
    )
    np.testing.assert_array_equal(
        first_detection.keypoints_xy_px,
        yolo_result.keypoints.xy[0],
    )
    np.testing.assert_array_equal(
        first_detection.keypoints_xy_norm,
        yolo_result.keypoints.xyn[0],
    )
    np.testing.assert_array_equal(
        first_detection.keypoints_conf,
        yolo_result.keypoints.conf[0],
    )

    assert second_detection.detection_idx == 1
    assert second_detection.track_id == 22
    assert second_detection.bbox_conf == pytest.approx(
        0.82
    )

    np.testing.assert_array_equal(
        second_detection.bbox_xyxy_px,
        yolo_result.boxes.xyxy[1],
    )
    np.testing.assert_array_equal(
        second_detection.bbox_xyxy_normalized,
        yolo_result.boxes.xyxyn[1],
    )
    np.testing.assert_array_equal(
        second_detection.keypoints_xy_px,
        yolo_result.keypoints.xy[1],
    )
    np.testing.assert_array_equal(
        second_detection.keypoints_xy_norm,
        yolo_result.keypoints.xyn[1],
    )
    np.testing.assert_array_equal(
        second_detection.keypoints_conf,
        yolo_result.keypoints.conf[1],
    )


def test_result_to_frame_detections_converts_tensor_like_values() -> None:
    yolo_result = _create_yolo_result(
        detection_count=1,
    )

    expected_confidence = (
        yolo_result.boxes.conf
    )
    expected_bbox_px = (
        yolo_result.boxes.xyxy
    )
    expected_bbox_normalized = (
        yolo_result.boxes.xyxyn
    )
    expected_keypoints_px = (
        yolo_result.keypoints.xy
    )
    expected_keypoints_normalized = (
        yolo_result.keypoints.xyn
    )
    expected_keypoint_confidence = (
        yolo_result.keypoints.conf
    )

    yolo_result.boxes.id = _FakeTensor(
        np.array(
            [15.0],
            dtype=np.float32,
        )
    )
    yolo_result.boxes.conf = _FakeTensor(
        expected_confidence
    )
    yolo_result.boxes.xyxy = _FakeTensor(
        expected_bbox_px
    )
    yolo_result.boxes.xyxyn = _FakeTensor(
        expected_bbox_normalized
    )
    yolo_result.keypoints.xy = _FakeTensor(
        expected_keypoints_px
    )
    yolo_result.keypoints.xyn = _FakeTensor(
        expected_keypoints_normalized
    )
    yolo_result.keypoints.conf = _FakeTensor(
        expected_keypoint_confidence
    )

    result = result_to_frame_detections(
        yolo_frame_result=yolo_result,
        frame_idx=0,
    )

    assert len(result.person_detections) == 1

    detection = result.person_detections[0]

    assert detection.track_id == 15

    np.testing.assert_array_equal(
        detection.bbox_xyxy_px,
        expected_bbox_px[0],
    )
    np.testing.assert_array_equal(
        detection.bbox_xyxy_normalized,
        expected_bbox_normalized[0],
    )
    np.testing.assert_array_equal(
        detection.keypoints_xy_px,
        expected_keypoints_px[0],
    )
    np.testing.assert_array_equal(
        detection.keypoints_xy_norm,
        expected_keypoints_normalized[0],
    )
    np.testing.assert_array_equal(
        detection.keypoints_conf,
        expected_keypoint_confidence[0],
    )


def test_result_to_frame_detections_uses_none_when_track_ids_are_missing() -> None:
    yolo_result = _create_yolo_result(
        detection_count=2,
        track_ids=None,
    )

    result = result_to_frame_detections(
        yolo_frame_result=yolo_result,
        frame_idx=0,
    )

    assert [
        detection.track_id
        for detection in result.person_detections
    ] == [
        None,
        None,
    ]


def test_result_to_frame_detections_flattens_track_id_array() -> None:
    yolo_result = _create_yolo_result(
        detection_count=2,
        track_ids=np.array(
            [
                [31.0],
                [42.0],
            ],
            dtype=np.float32,
        ),
    )

    result = result_to_frame_detections(
        yolo_frame_result=yolo_result,
        frame_idx=0,
    )

    assert [
        detection.track_id
        for detection in result.person_detections
    ] == [
        31,
        42,
    ]


def test_result_to_frame_detections_returns_empty_frame_when_boxes_are_none() -> None:
    yolo_result = SimpleNamespace(
        orig_shape=(720, 1280),
        boxes=None,
        keypoints=None,
    )

    result = result_to_frame_detections(
        yolo_frame_result=yolo_result,
        frame_idx=3,
    )

    assert result == FrameDetections(
        person_detections=[],
        frame_idx=3,
        frame_shape_hw=(720, 1280),
    )


def test_result_to_frame_detections_returns_empty_frame_when_boxes_are_empty() -> None:
    yolo_result = _create_yolo_result(
        detection_count=0,
    )

    result = result_to_frame_detections(
        yolo_frame_result=yolo_result,
        frame_idx=2,
    )

    assert result == FrameDetections(
        person_detections=[],
        frame_idx=2,
        frame_shape_hw=(720, 1280),
    )


def test_result_to_frame_detections_rejects_boxes_without_pose_keypoints() -> None:
    yolo_result = _create_yolo_result(
        detection_count=1,
    )
    yolo_result.keypoints = None

    with pytest.raises(
        ValueError,
        match=(
            "Frame 6 has boxes but no keypoints. "
            "Expected YOLO pose model output."
        ),
    ):
        result_to_frame_detections(
            yolo_frame_result=yolo_result,
            frame_idx=6,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "expected_value_name",
    ),
    [
        (
            "bbox_confidence",
            "bbox confidence scores",
        ),
        (
            "bbox_pixel_coordinates",
            "bbox absolute pixel coordinates",
        ),
        (
            "bbox_normalized_coordinates",
            "bbox normalized coordinates",
        ),
        (
            "raw_keypoints",
            "raw keypoints",
        ),
        (
            "normalized_keypoints",
            "normalised keypoints",
        ),
        (
            "keypoint_confidence",
            "keypoint confidence scores",
        ),
    ],
)
def test_result_to_frame_detections_rejects_missing_required_field(
    field_name: str,
    expected_value_name: str,
) -> None:
    yolo_result = _create_yolo_result(
        detection_count=1,
    )

    if field_name == "bbox_confidence":
        yolo_result.boxes.conf = None

    elif field_name == "bbox_pixel_coordinates":
        yolo_result.boxes.xyxy = None

    elif field_name == "bbox_normalized_coordinates":
        yolo_result.boxes.xyxyn = None

    elif field_name == "raw_keypoints":
        yolo_result.keypoints.xy = None

    elif field_name == "normalized_keypoints":
        yolo_result.keypoints.xyn = None

    elif field_name == "keypoint_confidence":
        yolo_result.keypoints.conf = None

    with pytest.raises(
        ValueError,
        match=(
            f"Missing {expected_value_name} "
            "in frame 8"
        ),
    ):
        result_to_frame_detections(
            yolo_frame_result=yolo_result,
            frame_idx=8,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "expected_value_name",
    ),
    [
        (
            "bbox_confidence",
            "bbox confidence scores",
        ),
        (
            "bbox_normalized_coordinates",
            "bbox normalized coordinates",
        ),
        (
            "raw_keypoints",
            "raw keypoints",
        ),
        (
            "normalized_keypoints",
            "normalised keypoints",
        ),
        (
            "keypoint_confidence",
            "keypoint confidence scores",
        ),
    ],
)
def test_result_to_frame_detections_rejects_mismatched_detection_count(
    field_name: str,
    expected_value_name: str,
) -> None:
    yolo_result = _create_yolo_result(
        detection_count=2,
    )

    if field_name == "bbox_confidence":
        yolo_result.boxes.conf = (
            yolo_result.boxes.conf[:1]
        )

    elif field_name == "bbox_normalized_coordinates":
        yolo_result.boxes.xyxyn = (
            yolo_result.boxes.xyxyn[:1]
        )

    elif field_name == "raw_keypoints":
        yolo_result.keypoints.xy = (
            yolo_result.keypoints.xy[:1]
        )

    elif field_name == "normalized_keypoints":
        yolo_result.keypoints.xyn = (
            yolo_result.keypoints.xyn[:1]
        )

    elif field_name == "keypoint_confidence":
        yolo_result.keypoints.conf = (
            yolo_result.keypoints.conf[:1]
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Frame 5: {expected_value_name} "
            "has 1 detections, expected 2."
        ),
    ):
        result_to_frame_detections(
            yolo_frame_result=yolo_result,
            frame_idx=5,
        )


def test_result_to_frame_detections_rejects_mismatched_track_id_count() -> None:
    yolo_result = _create_yolo_result(
        detection_count=2,
        track_ids=np.array(
            [11.0],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Frame 7 has 2 detections but "
            "1 track IDs."
        ),
    ):
        result_to_frame_detections(
            yolo_frame_result=yolo_result,
            frame_idx=7,
        )


def test_collect_clip_detections_converts_each_frame_in_order(
    mocker,
) -> None:
    first_yolo_result = object()
    second_yolo_result = object()
    third_yolo_result = object()

    first_frame = FrameDetections(
        person_detections=[],
        frame_idx=0,
        frame_shape_hw=(720, 1280),
    )
    second_frame = FrameDetections(
        person_detections=[],
        frame_idx=1,
        frame_shape_hw=(720, 1280),
    )
    third_frame = FrameDetections(
        person_detections=[],
        frame_idx=2,
        frame_shape_hw=(720, 1280),
    )

    result_to_frame_mock = mocker.patch(
        f"{RESULTS_ADAPTER_MODULE_PATH}."
        "result_to_frame_detections",
        side_effect=[
            first_frame,
            second_frame,
            third_frame,
        ],
    )

    def yolo_output_iterator():
        yield first_yolo_result
        yield second_yolo_result
        yield third_yolo_result

    result = collect_clip_detections(
        clip_id="clip_42",
        yolo_clip_output=(
            yolo_output_iterator()
        ),
    )

    assert isinstance(
        result,
        ClipDetections,
    )
    assert result.clip_id == "clip_42"
    assert result.frame_detections == [
        first_frame,
        second_frame,
        third_frame,
    ]

    assert (
        result_to_frame_mock.call_args_list
        == [
            call(
                yolo_frame_result=(
                    first_yolo_result
                ),
                frame_idx=0,
            ),
            call(
                yolo_frame_result=(
                    second_yolo_result
                ),
                frame_idx=1,
            ),
            call(
                yolo_frame_result=(
                    third_yolo_result
                ),
                frame_idx=2,
            ),
        ]
    )


def test_collect_clip_detections_returns_empty_clip_for_empty_iterator() -> None:
    result = collect_clip_detections(
        clip_id="empty_clip",
        yolo_clip_output=iter(()),
    )

    assert result == ClipDetections(
        frame_detections=[],
        clip_id="empty_clip",
    )


def test_collect_clip_detections_propagates_frame_conversion_failure(
    mocker,
) -> None:
    conversion_error = ValueError(
        "Invalid YOLO frame"
    )

    result_to_frame_mock = mocker.patch(
        f"{RESULTS_ADAPTER_MODULE_PATH}."
        "result_to_frame_detections",
        side_effect=conversion_error,
    )

    yolo_result = object()

    with pytest.raises(
        ValueError,
        match="Invalid YOLO frame",
    ) as exception_info:
        collect_clip_detections(
            clip_id="clip_0",
            yolo_clip_output=iter(
                [yolo_result]
            ),
        )

    assert (
        exception_info.value
        is conversion_error
    )

    result_to_frame_mock.assert_called_once_with(
        yolo_frame_result=yolo_result,
        frame_idx=0,
    )
