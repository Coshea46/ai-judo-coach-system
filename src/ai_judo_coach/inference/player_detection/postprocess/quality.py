"""
This file contains functions for assessing whether
interpolated two-player pose sequences contain enough
resolved pose data for downstream use.
"""

from dataclasses import dataclass

import numpy as np

from ai_judo_coach.inference.inference_schemas import (
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
)
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


@dataclass(frozen=True, slots=True)
class PoseSequenceQualityReport:
    accepted: bool
    rejection_reasons: tuple[str, ...]

    player_a_unusable_frame_fraction: float
    player_b_unusable_frame_fraction: float

    player_a_longest_unusable_gap: int
    player_b_longest_unusable_gap: int

    both_players_unusable_fraction: float


def assess_pose_sequence_quality(
    clip_player_pose_sequences: TwoPlayerPoseSequences,
    config: PlayerDetectionConfig,
) -> PoseSequenceQualityReport:
    """
    Assesses whether interpolated player pose sequences
    contain enough resolved data for downstream use.

    A frame is usable for a player if it contains at least
    config.min_resolved_keypoints_per_usable_frame keypoints
    with finite x and y coordinates.

    A sequence is rejected if either player:
    - exceeds the maximum unusable-frame fraction
    - exceeds the maximum consecutive unusable-frame gap
    """

    player_a_sequence = (
        clip_player_pose_sequences.player_a_pose_sequence
    )

    player_b_sequence = (
        clip_player_pose_sequences.player_b_pose_sequence
    )

    _validate_pose_sequence_shapes(
        player_a_sequence=player_a_sequence,
        player_b_sequence=player_b_sequence,
    )

    player_a_unusable_mask = _get_unusable_frame_mask(
        player_pose_sequence=player_a_sequence,
        min_resolved_keypoints=(
            config.min_resolved_keypoints_per_usable_frame
        ),
    )

    player_b_unusable_mask = _get_unusable_frame_mask(
        player_pose_sequence=player_b_sequence,
        min_resolved_keypoints=(
            config.min_resolved_keypoints_per_usable_frame
        ),
    )

    player_a_unusable_fraction = _boolean_mask_fraction(
        boolean_mask=player_a_unusable_mask
    )

    player_b_unusable_fraction = _boolean_mask_fraction(
        boolean_mask=player_b_unusable_mask
    )

    player_a_longest_unusable_gap = _longest_true_run(
        boolean_mask=player_a_unusable_mask
    )

    player_b_longest_unusable_gap = _longest_true_run(
        boolean_mask=player_b_unusable_mask
    )

    both_players_unusable_mask = (
        player_a_unusable_mask
        & player_b_unusable_mask
    )

    both_players_unusable_fraction = _boolean_mask_fraction(
        boolean_mask=both_players_unusable_mask
    )

    rejection_reasons: list[str] = []

    if (
        player_a_unusable_fraction
        > config.max_unusable_frame_fraction_per_player
    ):
        rejection_reasons.append(
            "player_a_unusable_frame_fraction_exceeded"
        )

    if (
        player_b_unusable_fraction
        > config.max_unusable_frame_fraction_per_player
    ):
        rejection_reasons.append(
            "player_b_unusable_frame_fraction_exceeded"
        )

    if (
        player_a_longest_unusable_gap
        > config.max_consecutive_unusable_frames_per_player
    ):
        rejection_reasons.append(
            "player_a_max_consecutive_unusable_frames_exceeded"
        )

    if (
        player_b_longest_unusable_gap
        > config.max_consecutive_unusable_frames_per_player
    ):
        rejection_reasons.append(
            "player_b_max_consecutive_unusable_frames_exceeded"
        )

    accepted = len(rejection_reasons) == 0

    return PoseSequenceQualityReport(
        accepted=accepted,
        rejection_reasons=tuple(rejection_reasons),
        player_a_unusable_frame_fraction=float(
            player_a_unusable_fraction
        ),
        player_b_unusable_frame_fraction=float(
            player_b_unusable_fraction
        ),
        player_a_longest_unusable_gap=(
            player_a_longest_unusable_gap
        ),
        player_b_longest_unusable_gap=(
            player_b_longest_unusable_gap
        ),
        both_players_unusable_fraction=float(
            both_players_unusable_fraction
        ),
    )


def _get_unusable_frame_mask(
    player_pose_sequence: PlayerPoseSequence,
    min_resolved_keypoints: int,
) -> np.ndarray:
    """
    Returns a boolean mask with shape [T].

    True means the player does not have enough resolved
    keypoints for the frame to be considered usable.
    """

    keypoints_xy_norm = player_pose_sequence.keypoints_xy_norm

    resolved_keypoint_mask = np.all(
        np.isfinite(keypoints_xy_norm),
        axis=2,
    )

    resolved_keypoint_count_per_frame = np.sum(
        resolved_keypoint_mask,
        axis=1,
    )

    usable_frame_mask = (
        resolved_keypoint_count_per_frame
        >= min_resolved_keypoints
    )

    return ~usable_frame_mask


def _boolean_mask_fraction(
    boolean_mask: np.ndarray,
) -> float:
    """
    Returns the fraction of values in a boolean mask
    that are True.
    """

    if len(boolean_mask) == 0:
        return 0.0

    return float(np.mean(boolean_mask))


def _longest_true_run(
    boolean_mask: np.ndarray,
) -> int:
    """
    Returns the length of the longest consecutive run
    of True values in a one-dimensional boolean mask.
    """

    longest_run = 0
    current_run = 0

    for value in boolean_mask:
        if value:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return longest_run


def _validate_pose_sequence_shapes(
    player_a_sequence: PlayerPoseSequence,
    player_b_sequence: PlayerPoseSequence,
) -> None:
    """
    Validates the coordinate-array shapes required
    for pose sequence quality assessment.
    """

    player_a_shape = player_a_sequence.keypoints_xy_norm.shape
    player_b_shape = player_b_sequence.keypoints_xy_norm.shape

    if (
        len(player_a_shape) != 3
        or player_a_shape[1:] != (17, 2)
    ):
        raise ValueError(
            "player_a keypoints_xy_norm must have shape "
            f"[T, 17, 2], got {player_a_shape}."
        )

    if (
        len(player_b_shape) != 3
        or player_b_shape[1:] != (17, 2)
    ):
        raise ValueError(
            "player_b keypoints_xy_norm must have shape "
            f"[T, 17, 2], got {player_b_shape}."
        )

    if player_a_shape[0] != player_b_shape[0]:
        raise ValueError(
            "Player A and player B pose sequences must "
            "contain the same number of frames."
        )

    if player_a_shape[0] == 0:
        raise ValueError(
            "Pose sequences must contain at least one frame."
        )
