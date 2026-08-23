import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PersonDetection,
)
from ai_judo_coach.inference.inference_schemas import (
    keypoints as kp,
)
from ai_judo_coach.inference.player_detection.features.pose import (
    average_body_length,
    mean_keypoint_confidence,
    visible_keypoint_count,
    visible_keypoint_fraction,
    visible_keypoint_mask,
)


def _create_person_detection(
    keypoints_xy_norm: np.ndarray | None = None,
    keypoints_conf: np.ndarray | None = None,
) -> PersonDetection:
    """Create one person detection for pose-feature tests."""

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
        bbox_xyxy_px=np.array(
            [10.0, 20.0, 50.0, 80.0],
            dtype=np.float32,
        ),
        bbox_xyxy_normalized=np.array(
            [0.1, 0.2, 0.5, 0.8],
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


def _create_body_keypoints() -> tuple[np.ndarray, np.ndarray]:
    """
    Create computable left and right body measurements.

    Left:
        torso = 0.3
        leg = 0.5

    Right:
        torso = 0.3
        leg = 0.3
    """

    keypoints = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    confidence = np.zeros(
        17,
        dtype=np.float32,
    )

    keypoints[kp.LEFT_SHOULDER] = [0.2, 0.1]
    keypoints[kp.LEFT_HIP] = [0.2, 0.4]
    keypoints[kp.LEFT_KNEE] = [0.2, 0.7]
    keypoints[kp.LEFT_ANKLE] = [0.2, 0.9]

    keypoints[kp.RIGHT_SHOULDER] = [0.8, 0.2]
    keypoints[kp.RIGHT_HIP] = [0.8, 0.5]
    keypoints[kp.RIGHT_KNEE] = [0.8, 0.7]
    keypoints[kp.RIGHT_ANKLE] = [0.8, 0.8]

    body_indices = [
        kp.LEFT_SHOULDER,
        kp.LEFT_HIP,
        kp.LEFT_KNEE,
        kp.LEFT_ANKLE,
        kp.RIGHT_SHOULDER,
        kp.RIGHT_HIP,
        kp.RIGHT_KNEE,
        kp.RIGHT_ANKLE,
    ]

    confidence[body_indices] = 0.9

    return keypoints, confidence


def test_mean_keypoint_confidence_returns_mean_of_finite_values() -> None:
    confidence = np.array(
        [
            0.1,
            0.3,
            np.nan,
            np.inf,
            -np.inf,
            0.8,
            0.4,
            0.2,
            0.6,
            0.7,
            0.5,
            0.9,
            1.0,
            0.0,
            0.25,
            0.75,
            0.45,
        ],
        dtype=np.float32,
    )

    person_detection = _create_person_detection(
        keypoints_conf=confidence,
    )

    result = mean_keypoint_confidence(
        player_detection=person_detection,
    )

    expected = np.mean(
        confidence[np.isfinite(confidence)]
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        float(expected)
    )


def test_mean_keypoint_confidence_returns_zero_without_finite_values() -> None:
    confidence = np.full(
        17,
        np.nan,
        dtype=np.float32,
    )
    confidence[0] = np.inf
    confidence[1] = -np.inf

    person_detection = _create_person_detection(
        keypoints_conf=confidence,
    )

    result = mean_keypoint_confidence(
        player_detection=person_detection,
    )

    assert result == pytest.approx(0.0)


def test_visible_keypoint_mask_identifies_usable_keypoints() -> None:
    keypoints = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    confidence = np.zeros(
        17,
        dtype=np.float32,
    )

    # Visible: finite, non-zero, and above threshold.
    keypoints[0] = [0.2, 0.3]
    confidence[0] = 0.9

    # Visible: confidence threshold is inclusive.
    keypoints[1] = [0.4, 0.5]
    confidence[1] = 0.5

    # Visible: only one coordinate needs to be non-zero.
    keypoints[2] = [0.0, 0.6]
    confidence[2] = 0.8

    # Not visible: below confidence threshold.
    keypoints[3] = [0.7, 0.8]
    confidence[3] = 0.49

    # Not visible: both coordinates are zero.
    keypoints[4] = [0.0, 0.0]
    confidence[4] = 0.9

    # Not visible: non-finite coordinate.
    keypoints[5] = [np.nan, 0.5]
    confidence[5] = 0.9

    keypoints[6] = [np.inf, 0.5]
    confidence[6] = 0.9

    # Not visible: non-finite confidence.
    keypoints[7] = [0.2, 0.4]
    confidence[7] = np.nan

    keypoints[8] = [0.2, 0.4]
    confidence[8] = np.inf

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = visible_keypoint_mask(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    expected = np.zeros(
        17,
        dtype=bool,
    )
    expected[[0, 1, 2]] = True

    assert result.shape == (17,)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_visible_keypoint_count_returns_number_of_usable_keypoints() -> None:
    keypoints = np.zeros(
        (17, 2),
        dtype=np.float32,
    )
    confidence = np.zeros(
        17,
        dtype=np.float32,
    )

    keypoints[0] = [0.1, 0.2]
    keypoints[4] = [0.3, 0.4]
    keypoints[12] = [0.5, 0.6]

    confidence[[0, 4, 12]] = [
        0.9,
        0.8,
        0.7,
    ]

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = visible_keypoint_count(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    assert isinstance(result, int)
    assert result == 3


def test_visible_keypoint_fraction_returns_fraction_of_coco_keypoints() -> None:
    keypoints = np.full(
        (17, 2),
        0.5,
        dtype=np.float32,
    )
    confidence = np.zeros(
        17,
        dtype=np.float32,
    )
    confidence[:8] = 0.9

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = visible_keypoint_fraction(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        8.0 / 17.0
    )


def test_visible_keypoint_fraction_returns_zero_without_usable_keypoints() -> None:
    person_detection = _create_person_detection()

    result = visible_keypoint_fraction(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.0)


def test_visible_keypoint_fraction_returns_one_when_all_keypoints_are_usable() -> None:
    person_detection = _create_person_detection(
        keypoints_xy_norm=np.full(
            (17, 2),
            0.5,
            dtype=np.float32,
        ),
        keypoints_conf=np.ones(
            17,
            dtype=np.float32,
        ),
    )

    result = visible_keypoint_fraction(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "min_keypoint_confidence",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_visible_keypoint_mask_rejects_invalid_confidence_threshold(
    min_keypoint_confidence: float,
) -> None:
    person_detection = _create_person_detection()

    with pytest.raises(
        ValueError,
        match=(
            "min_keypoint_confidence must be "
            r"in \[0, 1\]"
        ),
    ):
        visible_keypoint_mask(
            player_detection=person_detection,
            min_keypoint_confidence=(
                min_keypoint_confidence
            ),
        )


@pytest.mark.parametrize(
    "min_keypoint_confidence",
    [
        0.0,
        1.0,
    ],
)
def test_visible_keypoint_mask_accepts_threshold_boundaries(
    min_keypoint_confidence: float,
) -> None:
    keypoints = np.full(
        (17, 2),
        0.5,
        dtype=np.float32,
    )
    confidence = np.full(
        17,
        min_keypoint_confidence,
        dtype=np.float32,
    )

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = visible_keypoint_mask(
        player_detection=person_detection,
        min_keypoint_confidence=(
            min_keypoint_confidence
        ),
    )

    assert np.all(result)


def test_average_body_length_averages_both_body_sides() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Mean leg length: (0.5 + 0.3) / 2 = 0.4
    # Mean torso length: (0.3 + 0.3) / 2 = 0.3
    assert result == pytest.approx(
        0.7,
        abs=1e-6,
    )


def test_average_body_length_uses_single_computable_side() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    confidence[
        [
            kp.RIGHT_SHOULDER,
            kp.RIGHT_HIP,
            kp.RIGHT_KNEE,
            kp.RIGHT_ANKLE,
        ]
    ] = 0.0

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Left leg = 0.5 and left torso = 0.3.
    assert result == pytest.approx(
        0.8,
        abs=1e-6,
    )


def test_average_body_length_computes_torso_when_leg_is_unavailable() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    confidence[
        [
            kp.LEFT_KNEE,
            kp.RIGHT_KNEE,
        ]
    ] = 0.0

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Neither leg is computable, but both torso sides are 0.3.
    assert result == pytest.approx(
        0.3,
        abs=1e-6,
    )


def test_average_body_length_computes_leg_when_torso_is_unavailable() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    confidence[
        [
            kp.LEFT_SHOULDER,
            kp.RIGHT_SHOULDER,
        ]
    ] = 0.0

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Mean leg length is (0.5 + 0.3) / 2.
    assert result == pytest.approx(
        0.4,
        abs=1e-6,
    )


def test_average_body_length_returns_zero_without_computable_limbs() -> None:
    person_detection = _create_person_detection()

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "unusable_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_average_body_length_excludes_side_with_non_finite_joint(
    unusable_value: float,
) -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    keypoints[kp.LEFT_HIP, 0] = unusable_value

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Only the right leg and torso remain computable.
    assert result == pytest.approx(
        0.6,
        abs=1e-6,
    )


def test_average_body_length_excludes_side_with_zero_coordinate_pair() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    keypoints[kp.LEFT_HIP] = [0.0, 0.0]

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Only the right leg and torso remain computable.
    assert result == pytest.approx(
        0.6,
        abs=1e-6,
    )


def test_average_body_length_excludes_side_below_confidence_threshold() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    confidence[kp.LEFT_HIP] = 0.49

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    # Only the right leg and torso remain computable.
    assert result == pytest.approx(
        0.6,
        abs=1e-6,
    )


def test_average_body_length_accepts_confidence_at_threshold() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )
    confidence[confidence > 0.0] = 0.5

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    result = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=0.5,
    )

    assert result == pytest.approx(
        0.7,
        abs=1e-6,
    )


def test_visible_keypoint_mask_rejects_malformed_normalized_keypoints() -> None:
    person_detection = _create_person_detection()

    person_detection.keypoints_xy_norm = np.zeros(
        (16, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected keypoints_xy_norm to have "
            r"shape \(17, 2\)"
        ),
    ):
        visible_keypoint_mask(
            player_detection=person_detection,
            min_keypoint_confidence=0.5,
        )


def test_mean_keypoint_confidence_rejects_malformed_confidence_array() -> None:
    person_detection = _create_person_detection()

    person_detection.keypoints_conf = np.zeros(
        16,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected keypoints_conf to have "
            r"shape \(17,\)"
        ),
    ):
        mean_keypoint_confidence(
            player_detection=person_detection,
        )


def test_average_body_length_rejects_invalid_confidence_threshold() -> None:
    keypoints, confidence = (
        _create_body_keypoints()
    )

    person_detection = _create_person_detection(
        keypoints_xy_norm=keypoints,
        keypoints_conf=confidence,
    )

    with pytest.raises(
        ValueError,
        match=(
            "min_keypoint_confidence must be "
            r"in \[0, 1\]"
        ),
    ):
        average_body_length(
            player_detection=person_detection,
            min_keypoint_confidence=1.1,
        )
