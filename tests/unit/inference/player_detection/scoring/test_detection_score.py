import math

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.scoring.detection_score import (
    detection_score,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


DETECTION_SCORE_MODULE_PATH = (
    "ai_judo_coach.inference.player_detection."
    "scoring.detection_score"
)


def _create_person_detection(
    bbox_confidence: float = 0.8,
) -> PersonDetection:
    """Create one person detection for scoring tests."""

    return PersonDetection(
        detection_idx=0,
        track_id=1,
        bbox_xyxy_px=np.array(
            [10.0, 20.0, 50.0, 80.0],
            dtype=np.float32,
        ),
        bbox_xyxy_normalized=np.array(
            [0.1, 0.2, 0.5, 0.8],
            dtype=np.float32,
        ),
        bbox_conf=bbox_confidence,
        keypoints_xy_px=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_xy_norm=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_conf=np.zeros(
            17,
            dtype=np.float32,
        ),
    )


def test_detection_score_returns_normalized_weighted_sum(
    mocker,
) -> None:
    person_detection = _create_person_detection(
        bbox_confidence=0.8,
    )

    distance_mock = mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=math.sqrt(0.125),
    )
    confidence_mock = mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.6,
    )
    body_length_mock = mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.7,
    )

    config = PlayerDetectionConfig(
        keypoint_confidence_threshold=0.3,
        bbox_confidence_weight=1.0,
        bbox_center_closeness_weight=2.0,
        mean_keypoint_confidence_weight=3.0,
        pose_size_weight=4.0,
        max_expected_normalized_body_length=1.0,
    )

    result = detection_score(
        person_detection=person_detection,
        config=config,
    )

    # Distance sqrt(0.125) is half the maximum possible distance,
    # so the centre-closeness score is 0.5.
    expected_score = (
        (0.8 * 1.0)
        + (0.5 * 2.0)
        + (0.6 * 3.0)
        + (0.7 * 4.0)
    ) / 10.0

    assert isinstance(result, float)
    assert result == pytest.approx(
        expected_score
    )

    distance_mock.assert_called_once_with(
        person_detection=person_detection,
    )
    confidence_mock.assert_called_once_with(
        player_detection=person_detection,
    )
    body_length_mock.assert_called_once_with(
        player_detection=person_detection,
        min_keypoint_confidence=0.3,
    )


def test_detection_score_uses_bbox_confidence_value(
    mocker,
) -> None:
    person_detection = _create_person_detection(
        bbox_confidence=0.73,
    )

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.0,
    )

    result = detection_score(
        person_detection=person_detection,
        config=PlayerDetectionConfig(
            bbox_confidence_weight=1.0,
            bbox_center_closeness_weight=0.0,
            mean_keypoint_confidence_weight=0.0,
            pose_size_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.73)


def test_detection_score_uses_mean_keypoint_confidence(
    mocker,
) -> None:
    person_detection = _create_person_detection()

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=0.0,
    )
    confidence_mock = mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.64,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.0,
    )

    result = detection_score(
        person_detection=person_detection,
        config=PlayerDetectionConfig(
            bbox_confidence_weight=0.0,
            bbox_center_closeness_weight=0.0,
            mean_keypoint_confidence_weight=1.0,
            pose_size_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.64)
    confidence_mock.assert_called_once_with(
        player_detection=person_detection,
    )


@pytest.mark.parametrize(
    (
        "normalized_distance",
        "expected_score",
    ),
    [
        (0.0, 1.0),
        (math.sqrt(0.125), 0.5),
        (math.sqrt(0.5), 0.0),
        (1.0, 0.0),
        (-1.0, 1.0),
    ],
)
def test_detection_score_converts_center_distance_to_closeness(
    mocker,
    normalized_distance: float,
    expected_score: float,
) -> None:
    person_detection = _create_person_detection()

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=normalized_distance,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.0,
    )

    result = detection_score(
        person_detection=person_detection,
        config=PlayerDetectionConfig(
            bbox_confidence_weight=0.0,
            bbox_center_closeness_weight=1.0,
            mean_keypoint_confidence_weight=0.0,
            pose_size_weight=0.0,
        ),
    )

    assert result == pytest.approx(
        expected_score
    )


@pytest.mark.parametrize(
    (
        "pose_size",
        "maximum_expected_length",
        "expected_score",
    ),
    [
        (0.0, 1.0, 0.0),
        (0.4, 1.0, 0.4),
        (0.5, 2.0, 0.25),
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (-0.5, 1.0, 0.0),
    ],
)
def test_detection_score_normalizes_and_clamps_pose_size(
    mocker,
    pose_size: float,
    maximum_expected_length: float,
    expected_score: float,
) -> None:
    person_detection = _create_person_detection()

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.0,
    )
    body_length_mock = mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=pose_size,
    )

    config = PlayerDetectionConfig(
        keypoint_confidence_threshold=0.45,
        bbox_confidence_weight=0.0,
        bbox_center_closeness_weight=0.0,
        mean_keypoint_confidence_weight=0.0,
        pose_size_weight=1.0,
        max_expected_normalized_body_length=(
            maximum_expected_length
        ),
    )

    result = detection_score(
        person_detection=person_detection,
        config=config,
    )

    assert result == pytest.approx(
        expected_score
    )

    body_length_mock.assert_called_once_with(
        player_detection=person_detection,
        min_keypoint_confidence=0.45,
    )


@pytest.mark.parametrize(
    "maximum_expected_length",
    [
        0.0,
        -1.0,
    ],
)
def test_detection_score_returns_zero_pose_size_score_for_non_positive_maximum(
    mocker,
    maximum_expected_length: float,
) -> None:
    person_detection = _create_person_detection()

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.8,
    )

    result = detection_score(
        person_detection=person_detection,
        config=PlayerDetectionConfig(
            bbox_confidence_weight=0.0,
            bbox_center_closeness_weight=0.0,
            mean_keypoint_confidence_weight=0.0,
            pose_size_weight=1.0,
            max_expected_normalized_body_length=(
                maximum_expected_length
            ),
        ),
    )

    assert result == pytest.approx(0.0)


def test_detection_score_returns_zero_when_all_weights_are_zero(
    mocker,
) -> None:
    person_detection = _create_person_detection(
        bbox_confidence=0.9,
    )

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.9,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.9,
    )

    result = detection_score(
        person_detection=person_detection,
        config=PlayerDetectionConfig(
            bbox_confidence_weight=0.0,
            bbox_center_closeness_weight=0.0,
            mean_keypoint_confidence_weight=0.0,
            pose_size_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.0)


def test_detection_score_ignores_features_with_zero_weights(
    mocker,
) -> None:
    person_detection = _create_person_detection(
        bbox_confidence=0.25,
    )

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=math.sqrt(0.5),
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=1.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=1.0,
    )

    result = detection_score(
        person_detection=person_detection,
        config=PlayerDetectionConfig(
            bbox_confidence_weight=1.0,
            bbox_center_closeness_weight=0.0,
            mean_keypoint_confidence_weight=0.0,
            pose_size_weight=0.0,
        ),
    )

    assert result == pytest.approx(0.25)


def test_detection_score_with_default_config_is_normalized_weighted_average(
    mocker,
) -> None:
    person_detection = _create_person_detection(
        bbox_confidence=0.8,
    )

    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "normalized_bbox_distance_to_frame_center",
        return_value=0.0,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "mean_keypoint_confidence",
        return_value=0.6,
    )
    mocker.patch(
        f"{DETECTION_SCORE_MODULE_PATH}."
        "average_body_length",
        return_value=0.5,
    )

    config = PlayerDetectionConfig()

    result = detection_score(
        person_detection=person_detection,
        config=config,
    )

    expected_score = (
        (0.8 * 0.2)
        + (1.0 * 0.4)
        + (0.6 * 0.2)
        + (0.5 * 0.6)
    ) / (
        0.2
        + 0.4
        + 0.2
        + 0.6
    )

    assert result == pytest.approx(
        expected_score
    )
