"""
Pose/keypoint feature utilities for player detection.

This module should contain only derived measurements from a single
PersonDetection's keypoints.

COCO-17 keypoint format is assumed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ai_judo_coach.inference.inference_schemas import PersonDetection
from schemas import keypoints as kp


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


def mean_keypoint_confidence(player_detection: PersonDetection) -> float:
    """
    Return the mean keypoint confidence over finite confidence values.

    If no finite confidence values exist, returns 0.0.
    """
    confidence_scores = _keypoint_confidence_scores(player_detection)

    finite_mask = np.isfinite(confidence_scores)

    if not np.any(finite_mask):
        return 0.0

    return float(np.mean(confidence_scores[finite_mask]))


def visible_keypoint_mask(
    player_detection: PersonDetection,
    min_keypoint_confidence: float,
) -> BoolArray:
    """
    Return a boolean mask saying which keypoints are usable.

    A keypoint is usable if:
        - coordinates are finite
        - coordinates are not [0, 0]
        - confidence is finite
        - confidence >= min_keypoint_confidence

    Returns
    -------
    np.ndarray
        Shape [17], dtype bool.
    """
    normalized_keypoints = _normalized_keypoints(player_detection)
    confidence_scores = _keypoint_confidence_scores(player_detection)

    return _visible_keypoint_mask_from_arrays(
        normalized_keypoints=normalized_keypoints,
        keypoint_confidence_scores=confidence_scores,
        min_keypoint_confidence=min_keypoint_confidence,
    )


def visible_keypoint_count(
    player_detection: PersonDetection,
    min_keypoint_confidence: float,
) -> int:
    """
    Return the number of usable keypoints.
    """
    mask = visible_keypoint_mask(
        player_detection=player_detection,
        min_keypoint_confidence=min_keypoint_confidence,
    )

    return int(np.sum(mask))


def visible_keypoint_fraction(
    player_detection: PersonDetection,
    min_keypoint_confidence: float,
) -> float:
    """
    Return the fraction of COCO-17 keypoints that are usable.
    """
    return visible_keypoint_count(
        player_detection=player_detection,
        min_keypoint_confidence=min_keypoint_confidence,
    ) / 17.0


def average_body_length(
    player_detection: PersonDetection,
    min_keypoint_confidence: float,
) -> float:
    """
    Estimate body length using normalised keypoint coordinates.

    This computes:

        average leg length + average torso length

    Leg length is computed as:

        ankle-to-knee + knee-to-hip

    Torso length is computed as:

        shoulder-to-hip

    Left and right sides are averaged separately. If one side is missing, it is
    not included in the average.

    A keypoint is treated as missing if:
        - coordinate is NaN or infinite
        - coordinate is [0, 0]
        - confidence is NaN or infinite
        - confidence < min_keypoint_confidence

    Returns
    -------
    float
        Estimated body length in normalised-coordinate units.
    """
    normalized_keypoints = _normalized_keypoints(player_detection)
    confidence_scores = _keypoint_confidence_scores(player_detection)

    average_leg_length = _compute_average_leg_length(
        normalized_keypoints=normalized_keypoints,
        keypoint_confidence_scores=confidence_scores,
        min_keypoint_confidence=min_keypoint_confidence,
    )

    average_torso_length = _compute_average_torso_length(
        normalized_keypoints=normalized_keypoints,
        keypoint_confidence_scores=confidence_scores,
        min_keypoint_confidence=min_keypoint_confidence,
    )

    return average_leg_length + average_torso_length


def _normalized_keypoints(player_detection: PersonDetection) -> FloatArray:
    """
    Return normalised keypoints as float32 array with shape [17, 2].
    """
    normalized_keypoints = np.asarray(
        player_detection.keypoints_xy_norm,
        dtype=np.float32,
    )

    if normalized_keypoints.shape != (17, 2):
        raise ValueError(
            "Expected keypoints_xy_norm to have shape (17, 2), "
            f"got {normalized_keypoints.shape}"
        )

    return normalized_keypoints


def _keypoint_confidence_scores(player_detection: PersonDetection) -> FloatArray:
    """
    Return keypoint confidence scores as float32 array with shape [17].
    """
    confidence_scores = np.asarray(
        player_detection.keypoints_conf,
        dtype=np.float32,
    )

    if confidence_scores.shape != (17,):
        raise ValueError(
            "Expected keypoints_conf to have shape (17,), "
            f"got {confidence_scores.shape}"
        )

    return confidence_scores


def _visible_keypoint_mask_from_arrays(
    normalized_keypoints: np.ndarray,
    keypoint_confidence_scores: np.ndarray,
    min_keypoint_confidence: float,
) -> BoolArray:
    """
    Vectorised keypoint validity check.

    This helper assumes COCO-17 keypoints.
    """
    if not 0.0 <= min_keypoint_confidence <= 1.0:
        raise ValueError(
            "min_keypoint_confidence must be in [0, 1], "
            f"got {min_keypoint_confidence}"
        )

    normalized_keypoints = np.asarray(normalized_keypoints, dtype=np.float32)
    keypoint_confidence_scores = np.asarray(
        keypoint_confidence_scores,
        dtype=np.float32,
    )

    if normalized_keypoints.shape != (17, 2):
        raise ValueError(
            "Expected normalized_keypoints to have shape (17, 2), "
            f"got {normalized_keypoints.shape}"
        )

    if keypoint_confidence_scores.shape != (17,):
        raise ValueError(
            "Expected keypoint_confidence_scores to have shape (17,), "
            f"got {keypoint_confidence_scores.shape}"
        )

    coordinates_are_finite = np.all(np.isfinite(normalized_keypoints), axis=1)
    coordinates_are_not_zero = ~np.all(normalized_keypoints == 0.0, axis=1)

    confidence_is_finite = np.isfinite(keypoint_confidence_scores)
    confidence_is_high_enough = (
        keypoint_confidence_scores >= min_keypoint_confidence
    )

    return (
        coordinates_are_finite
        & coordinates_are_not_zero
        & confidence_is_finite
        & confidence_is_high_enough
    )


def _check_limb_computatable(
    normalized_keypoints: np.ndarray,
    keypoint_confidence_scores: np.ndarray,
    min_keypoint_confidence: float,
    *joints: int,
) -> bool:
    """
    Return True if all given joints are usable.

    Joint indices should align with COCO-17 pose format.
    """
    visible_mask = _visible_keypoint_mask_from_arrays(
        normalized_keypoints=normalized_keypoints,
        keypoint_confidence_scores=keypoint_confidence_scores,
        min_keypoint_confidence=min_keypoint_confidence,
    )

    for joint in joints:
        if joint is None:
            return False

        joint_idx = int(joint)

        if joint_idx < 0 or joint_idx >= visible_mask.shape[0]:
            return False

        if not bool(visible_mask[joint_idx]):
            return False

    return True


def _distance_between_two_keypoints(
    keypoint_a: np.ndarray,
    keypoint_b: np.ndarray,
) -> float:
    """
    Return Euclidean distance between two keypoint coordinates.
    """
    keypoint_a = np.asarray(keypoint_a, dtype=np.float32)
    keypoint_b = np.asarray(keypoint_b, dtype=np.float32)

    if keypoint_a.shape != (2,):
        raise ValueError(f"Expected keypoint_a shape (2,), got {keypoint_a.shape}")

    if keypoint_b.shape != (2,):
        raise ValueError(f"Expected keypoint_b shape (2,), got {keypoint_b.shape}")

    return float(np.linalg.norm(keypoint_a - keypoint_b))


def _compute_leg_length(
    ankle: np.ndarray,
    knee: np.ndarray,
    hip: np.ndarray,
) -> float:
    """
    Compute one leg length as ankle-to-knee plus knee-to-hip.
    """
    shin_length = _distance_between_two_keypoints(knee, ankle)
    thigh_length = _distance_between_two_keypoints(hip, knee)

    return shin_length + thigh_length


def _compute_average_leg_length(
    normalized_keypoints: np.ndarray,
    keypoint_confidence_scores: np.ndarray,
    min_keypoint_confidence: float,
) -> float:
    """
    Compute average length over computable left/right legs.

    If neither leg is computable, returns 0.0.
    """
    leg_lengths: list[float] = []

    if _check_limb_computatable(
        normalized_keypoints,
        keypoint_confidence_scores,
        min_keypoint_confidence,
        kp.LEFT_ANKLE,
        kp.LEFT_KNEE,
        kp.LEFT_HIP,
    ):
        left_leg_length = _compute_leg_length(
            ankle=normalized_keypoints[kp.LEFT_ANKLE],
            knee=normalized_keypoints[kp.LEFT_KNEE],
            hip=normalized_keypoints[kp.LEFT_HIP],
        )
        leg_lengths.append(left_leg_length)

    if _check_limb_computatable(
        normalized_keypoints,
        keypoint_confidence_scores,
        min_keypoint_confidence,
        kp.RIGHT_ANKLE,
        kp.RIGHT_KNEE,
        kp.RIGHT_HIP,
    ):
        right_leg_length = _compute_leg_length(
            ankle=normalized_keypoints[kp.RIGHT_ANKLE],
            knee=normalized_keypoints[kp.RIGHT_KNEE],
            hip=normalized_keypoints[kp.RIGHT_HIP],
        )
        leg_lengths.append(right_leg_length)

    if len(leg_lengths) == 0:
        return 0.0

    return float(np.mean(leg_lengths))


def _compute_average_torso_length(
    normalized_keypoints: np.ndarray,
    keypoint_confidence_scores: np.ndarray,
    min_keypoint_confidence: float,
) -> float:
    """
    Compute average length over computable left/right torso sides.

    If neither side is computable, returns 0.0.
    """
    torso_lengths: list[float] = []

    if _check_limb_computatable(
        normalized_keypoints,
        keypoint_confidence_scores,
        min_keypoint_confidence,
        kp.LEFT_SHOULDER,
        kp.LEFT_HIP,
    ):
        left_torso_length = _distance_between_two_keypoints(
            normalized_keypoints[kp.LEFT_SHOULDER],
            normalized_keypoints[kp.LEFT_HIP],
        )
        torso_lengths.append(left_torso_length)

    if _check_limb_computatable(
        normalized_keypoints,
        keypoint_confidence_scores,
        min_keypoint_confidence,
        kp.RIGHT_SHOULDER,
        kp.RIGHT_HIP,
    ):
        right_torso_length = _distance_between_two_keypoints(
            normalized_keypoints[kp.RIGHT_SHOULDER],
            normalized_keypoints[kp.RIGHT_HIP],
        )
        torso_lengths.append(right_torso_length)

    if len(torso_lengths) == 0:
        return 0.0

    return float(np.mean(torso_lengths))


__all__ = [
    "mean_keypoint_confidence",
    "visible_keypoint_mask",
    "visible_keypoint_count",
    "visible_keypoint_fraction",
    "average_body_length",
]
