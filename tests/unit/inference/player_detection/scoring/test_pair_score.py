import math

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.scoring.pair_score import (
    pair_score,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


PAIR_SCORE_MODULE_PATH = (
    "ai_judo_coach.inference.player_detection."
    "scoring.pair_score"
)


def _create_person_detection(
    detection_idx: int,
    bbox_xyxy_px: list[float] | None = None,
    bbox_xyxy_normalized: list[float] | None = None,
    keypoints_xy_norm: np.ndarray | None = None,
    keypoints_conf: np.ndarray | None = None,
) -> PersonDetection:
    """Create one person detection for pair-scoring tests."""

    if bbox_xyxy_px is None:
        bbox_xyxy_px = [
            10.0,
            20.0,
            50.0,
            80.0,
        ]

    if bbox_xyxy_normalized is None:
        bbox_xyxy_normalized = [
            0.1,
            0.2,
            0.5,
            0.8,
        ]

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
        detection_idx=detection_idx,
        track_id=detection_idx,
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


def test_pair_score_returns_normalized_weighted_sum(
    mocker,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    bbox_iou_mock = mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=0.4,
    )
    keypoint_proximity_mock = mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=0.7,
    )
    center_distance_mock = mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0) * 0.25,
    )

    config = PlayerDetectionConfig(
        keypoint_confidence_threshold=0.35,
        bbox_overlap_weight=1.0,
        average_keypoint_proximity_weight=2.0,
        pair_bbox_center_closeness_weight=3.0,
    )

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=config,
    )

    # A distance of 25% of the maximum gives a closeness of 0.75.
    expected_score = (
        (0.4 * 1.0)
        + (0.7 * 2.0)
        + (0.75 * 3.0)
    ) / 6.0

    assert isinstance(result, float)
    assert result == pytest.approx(
        expected_score
    )

    bbox_iou_mock.assert_called_once_with(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )
    keypoint_proximity_mock.assert_called_once_with(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.35,
    )
    center_distance_mock.assert_called_once_with(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
    )


def test_pair_score_uses_bbox_overlap_score(
    mocker,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=0.62,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=0.0,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0),
    )

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=PlayerDetectionConfig(
            bbox_overlap_weight=1.0,
            average_keypoint_proximity_weight=0.0,
            pair_bbox_center_closeness_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.62)


def test_pair_score_uses_average_keypoint_proximity(
    mocker,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=0.0,
    )
    proximity_mock = mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=0.73,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0),
    )

    config = PlayerDetectionConfig(
        keypoint_confidence_threshold=0.45,
        bbox_overlap_weight=0.0,
        average_keypoint_proximity_weight=1.0,
        pair_bbox_center_closeness_weight=0.0,
    )

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=config,
    )

    assert result == pytest.approx(0.73)

    proximity_mock.assert_called_once_with(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=0.45,
    )


@pytest.mark.parametrize(
    (
        "normalized_distance",
        "expected_closeness",
    ),
    [
        (
            0.0,
            1.0,
        ),
        (
            math.sqrt(2.0) * 0.25,
            0.75,
        ),
        (
            math.sqrt(2.0) * 0.5,
            0.5,
        ),
        (
            math.sqrt(2.0),
            0.0,
        ),
        (
            math.sqrt(2.0) * 2.0,
            0.0,
        ),
        (
            -1.0,
            1.0,
        ),
    ],
)
def test_pair_score_converts_bbox_center_distance_to_closeness(
    mocker,
    normalized_distance: float,
    expected_closeness: float,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=0.0,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=0.0,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=normalized_distance,
    )

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=PlayerDetectionConfig(
            bbox_overlap_weight=0.0,
            average_keypoint_proximity_weight=0.0,
            pair_bbox_center_closeness_weight=1.0,
        ),
    )

    assert result == pytest.approx(
        expected_closeness
    )


def test_pair_score_returns_zero_when_all_weights_are_zero(
    mocker,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=1.0,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=1.0,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=0.0,
    )

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=PlayerDetectionConfig(
            bbox_overlap_weight=0.0,
            average_keypoint_proximity_weight=0.0,
            pair_bbox_center_closeness_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.0)


def test_pair_score_ignores_features_with_zero_weights(
    mocker,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=0.4,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=1.0,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=0.0,
    )

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=PlayerDetectionConfig(
            bbox_overlap_weight=1.0,
            average_keypoint_proximity_weight=0.0,
            pair_bbox_center_closeness_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.4)


def test_pair_score_with_default_config_returns_expected_score(
    mocker,
) -> None:
    person_detection_a = _create_person_detection(
        detection_idx=0,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
    )

    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}.bbox_iou",
        return_value=0.5,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "average_keypoint_proximity",
        return_value=0.8,
    )
    mocker.patch(
        f"{PAIR_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0) * 0.25,
    )

    config = PlayerDetectionConfig()

    result = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=config,
    )

    expected_score = (
        (0.5 * 0.4)
        + (0.8 * 0.4)
        + (0.75 * 0.5)
    ) / (
        0.4
        + 0.4
        + 0.5
    )

    assert result == pytest.approx(
        expected_score
    )


def test_pair_score_is_identity_agnostic() -> None:
    player_a_keypoints = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    player_b_keypoints = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    player_a_confidence = np.zeros(
        17,
        dtype=np.float32,
    )
    player_b_confidence = np.zeros(
        17,
        dtype=np.float32,
    )

    player_a_keypoints[0] = [0.2, 0.3]
    player_b_keypoints[0] = [0.6, 0.7]
    player_a_confidence[0] = 0.9
    player_b_confidence[0] = 0.9

    person_detection_a = _create_person_detection(
        detection_idx=0,
        bbox_xyxy_px=[
            0.0,
            0.0,
            100.0,
            100.0,
        ],
        bbox_xyxy_normalized=[
            0.1,
            0.1,
            0.5,
            0.5,
        ],
        keypoints_xy_norm=player_a_keypoints,
        keypoints_conf=player_a_confidence,
    )
    person_detection_b = _create_person_detection(
        detection_idx=1,
        bbox_xyxy_px=[
            50.0,
            50.0,
            150.0,
            150.0,
        ],
        bbox_xyxy_normalized=[
            0.4,
            0.4,
            0.8,
            0.8,
        ],
        keypoints_xy_norm=player_b_keypoints,
        keypoints_conf=player_b_confidence,
    )

    config = PlayerDetectionConfig()

    result_ab = pair_score(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        config=config,
    )
    result_ba = pair_score(
        person_detection_a=person_detection_b,
        person_detection_b=person_detection_a,
        config=config,
    )

    assert result_ab == pytest.approx(
        result_ba
    )
    assert 0.0 <= result_ab <= 1.0
