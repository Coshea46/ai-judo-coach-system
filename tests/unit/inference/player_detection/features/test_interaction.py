import math

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.features.interaction import (
    average_keypoint_proximity,
    average_nearest_keypoint_distance,
    bbox_iou,
    distance_between_bbox_centers,
    normalized_distance_between_bbox_centers,
)


def _create_person_detection(
    bbox_xyxy_px: np.ndarray | list[float] = (
        10.0,
        20.0,
        50.0,
        80.0,
    ),
    bbox_xyxy_normalized: np.ndarray | list[float] = (
        0.1,
        0.2,
        0.5,
        0.8,
    ),
    keypoints_xy_norm: np.ndarray | None = None,
    keypoints_conf: np.ndarray | None = None,
) -> PersonDetection:
    """Create one person detection for interaction tests."""

    if keypoints_xy_norm is None:
        keypoints_xy_norm = np.zeros(
            (17, 2),
            dtype=np.float32,
        )

    if keypoints_conf is None:
        keypoints_conf = np.zeros(
            17,
            dtype=np.float32,
        )

    return PersonDetection(
        detection_idx=0,
        track_id=1,
        bbox_xyxy_px=np.asarray(
            bbox_xyxy_px,
            dtype=np.float32,
        ),
        bbox_xyxy_normalized=np.asarray(
            bbox_xyxy_normalized,
            dtype=np.float32,
        ),
        bbox_conf=0.9,
        keypoints_xy_px=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_xy_norm=np.asarray(
            keypoints_xy_norm,
            dtype=np.float32,
        ),
        keypoints_conf=np.asarray(
            keypoints_conf,
            dtype=np.float32,
        ),
    )


def _create_keypoint_arrays(
    visible_keypoints: list[tuple[float, float]],
    confidence: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create keypoint coordinate and confidence arrays."""

    keypoints_xy_norm = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    keypoints_conf = np.zeros(
        17,
        dtype=np.float32,
    )

    for keypoint_index, keypoint in enumerate(
        visible_keypoints
    ):
        keypoints_xy_norm[keypoint_index] = keypoint
        keypoints_conf[keypoint_index] = confidence

    return keypoints_xy_norm, keypoints_conf


def test_bbox_iou_returns_one_for_identical_boxes() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            10.0,
            20.0,
            50.0,
            80.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            10.0,
            20.0,
            50.0,
            80.0,
        ],
    )

    result = bbox_iou(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_bbox_iou_returns_expected_value_for_partial_overlap() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            0.0,
            0.0,
            10.0,
            10.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            5.0,
            5.0,
            15.0,
            15.0,
        ],
    )

    result = bbox_iou(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert result == pytest.approx(
        25.0 / 175.0
    )


def test_bbox_iou_returns_expected_value_when_one_box_contains_another() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            0.0,
            0.0,
            10.0,
            10.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            2.5,
            2.5,
            7.5,
            7.5,
        ],
    )

    result = bbox_iou(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert result == pytest.approx(0.25)


@pytest.mark.parametrize(
    (
        "bbox_a",
        "bbox_b",
    ),
    [
        (
            [0.0, 0.0, 5.0, 5.0],
            [10.0, 10.0, 15.0, 15.0],
        ),
        (
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 0.0, 10.0, 5.0],
        ),
        (
            [0.0, 0.0, 5.0, 5.0],
            [0.0, 5.0, 5.0, 10.0],
        ),
    ],
)
def test_bbox_iou_returns_zero_without_positive_area_overlap(
    bbox_a: list[float],
    bbox_b: list[float],
) -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=bbox_a,
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=bbox_b,
    )

    result = bbox_iou(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert result == pytest.approx(0.0)


def test_bbox_iou_returns_zero_when_union_area_is_zero() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            5.0,
            5.0,
            5.0,
            5.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            5.0,
            5.0,
            5.0,
            5.0,
        ],
    )

    result = bbox_iou(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert result == pytest.approx(0.0)


def test_bbox_iou_is_symmetric() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            0.0,
            0.0,
            10.0,
            10.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            5.0,
            5.0,
            15.0,
            15.0,
        ],
    )

    result_ab = bbox_iou(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )
    result_ba = bbox_iou(
        person_detection_a=person_detection_b,
        person_detection_b=person_detection_a,
    )

    assert result_ab == pytest.approx(result_ba)


def test_distance_between_bbox_centers_returns_pixel_distance() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            0.0,
            0.0,
            2.0,
            2.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            4.0,
            5.0,
            6.0,
            7.0,
        ],
    )

    result = distance_between_bbox_centers(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        math.sqrt(41.0)
    )


def test_distance_between_bbox_centers_returns_zero_for_same_center() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_px=[
            0.0,
            0.0,
            10.0,
            10.0,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_px=[
            2.0,
            2.0,
            8.0,
            8.0,
        ],
    )

    result = distance_between_bbox_centers(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert result == pytest.approx(0.0)


def test_normalized_distance_between_bbox_centers() -> None:
    person_detection_a = _create_person_detection(
        bbox_xyxy_normalized=[
            0.0,
            0.0,
            0.2,
            0.2,
        ],
    )
    person_detection_b = _create_person_detection(
        bbox_xyxy_normalized=[
            0.5,
            0.6,
            0.7,
            0.8,
        ],
    )

    result = normalized_distance_between_bbox_centers(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        math.sqrt(0.5**2 + 0.6**2)
    )


def test_average_nearest_keypoint_distance_for_one_keypoint_per_pose() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.1, 0.1),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.4, 0.5),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result is not None
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


def test_average_nearest_keypoint_distance_averages_both_directions() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.1, 0.1),
                (0.9, 0.1),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.1, 0.1),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.2)


def test_average_nearest_keypoint_distance_uses_confidence_threshold_inclusively() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.2, 0.2),
            ],
            confidence=0.5,
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.5, 0.6),
            ],
            confidence=0.5,
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.5)


def test_average_nearest_keypoint_distance_ignores_unusable_keypoints() -> None:
    player_a_keypoints = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    player_a_confidence = np.zeros(
        17,
        dtype=np.float32,
    )

    player_a_keypoints[0] = [0.2, 0.2]
    player_a_confidence[0] = 0.9

    # Below the confidence threshold.
    player_a_keypoints[1] = [0.9, 0.9]
    player_a_confidence[1] = 0.49

    # Zero coordinates are not visible.
    player_a_keypoints[2] = [0.0, 0.0]
    player_a_confidence[2] = 0.9

    # Non-finite coordinates are not visible.
    player_a_keypoints[3] = [np.nan, 0.5]
    player_a_confidence[3] = 0.9
    player_a_keypoints[4] = [np.inf, 0.5]
    player_a_confidence[4] = 0.9

    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.5, 0.6),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.5)


@pytest.mark.parametrize(
    "empty_player",
    [
        "player_a",
        "player_b",
        "both",
    ],
)
def test_average_nearest_keypoint_distance_returns_none_without_usable_keypoints(
    empty_player: str,
) -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=(
                []
                if empty_player in ("player_a", "both")
                else [(0.2, 0.2)]
            ),
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=(
                []
                if empty_player in ("player_b", "both")
                else [(0.5, 0.5)]
            ),
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result is None


def test_average_nearest_keypoint_distance_is_symmetric() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.1, 0.2),
                (0.4, 0.8),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.3, 0.4),
                (0.9, 0.7),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result_ab = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )
    result_ba = average_nearest_keypoint_distance(
        person_detection_a=person_detection_b,
        person_detection_b=person_detection_a,
        min_keypoint_confidence=0.5,
    )

    assert result_ab is not None
    assert result_ba == pytest.approx(result_ab)


def test_average_keypoint_proximity_is_one_for_identical_keypoints() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.25, 0.75),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.25, 0.75),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_keypoint_proximity(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_average_keypoint_proximity_returns_expected_score() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.0, 0.5),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.5, 0.5),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_keypoint_proximity(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    expected_proximity = (
        1.0
        - (0.5 / math.sqrt(2.0))
    )

    assert result == pytest.approx(
        expected_proximity
    )


def test_average_keypoint_proximity_is_zero_at_maximum_distance() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.0, 1.0),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (1.0, 0.0),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_keypoint_proximity(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.0)


def test_average_keypoint_proximity_clamps_negative_score_to_zero() -> None:
    player_a_keypoints, player_a_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (-1.0, -1.0),
            ],
        )
    )
    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (2.0, 2.0),
            ],
        )
    )

    person_detection_a = _create_person_detection(
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_keypoint_proximity(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.0)


def test_average_keypoint_proximity_returns_zero_without_usable_keypoints() -> None:
    person_detection_a = _create_person_detection()

    player_b_keypoints, player_b_confidence = (
        _create_keypoint_arrays(
            visible_keypoints=[
                (0.5, 0.5),
            ],
        )
    )
    person_detection_b = _create_person_detection(
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    result = average_keypoint_proximity(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.0)
