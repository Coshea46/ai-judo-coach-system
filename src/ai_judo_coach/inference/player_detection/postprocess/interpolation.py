"""
This file contains functions for interpolating
missing keypoints in two-player pose sequences.
"""

import numpy as np

from  ai_judo_coach.inference.inference_schemas import (
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
)
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


def interpolate_two_player_pose_sequences_in_place(
    clip_player_pose_sequences: TwoPlayerPoseSequences,
    config: PlayerDetectionConfig,
) -> None:
    """
    Interpolates eligible gaps in both player pose
    sequences by modifying their coordinate arrays
    in place.
    """

    _interpolate_single_player_sequence(
        player_pose_sequence=clip_player_pose_sequences.player_a_pose_sequence,
        config=config
    )

    _interpolate_single_player_sequence(
        player_pose_sequence=clip_player_pose_sequences.player_b_pose_sequence,
        config=config
    )


def _interpolate_single_player_sequence(
    player_pose_sequence: PlayerPoseSequence,
    config: PlayerDetectionConfig
) -> None:
    """
    Interpolates the pose sequence for a single player.
    """

    interp_norm_keypoint_sequences: list[np.ndarray] = []
    interp_raw_keypoint_sequences: list[np.ndarray] = []

    for keypoint_idx in range(17):
        normalized_joint_kp_sequence = (
            player_pose_sequence.keypoints_xy_norm[:, keypoint_idx, :]
        )

        raw_joint_kp_sequence = (
            player_pose_sequence.keypoints_xy_px[:, keypoint_idx, :]
        )

        joint_confidence_sequence = (
            player_pose_sequence.keypoints_conf[:, keypoint_idx]
        )

        missing_points_mask = _get_missing_keypoint_mask(
            norm_joint_keypoint_sequence=normalized_joint_kp_sequence,
            joint_confidence_sequence=joint_confidence_sequence,
            detection_missing_mask=player_pose_sequence.missing_mask,
            min_keypoint_confidence=config.keypoint_confidence_threshold
        )

        gaps_to_interpolate = _find_interpolatable_gaps(
            missing_points_mask=missing_points_mask,
            longest_gap_allowed=config.longest_gap_allowed
        )

        norm_interpolated_joint_kps = _interpolate_gaps(
            joint_keypoint_sequence=normalized_joint_kp_sequence,
            missing_points_mask=missing_points_mask,
            gaps_to_interpolate=gaps_to_interpolate
        )

        raw_interpolated_joint_kps = _interpolate_gaps(
            joint_keypoint_sequence=raw_joint_kp_sequence,
            missing_points_mask=missing_points_mask,
            gaps_to_interpolate=gaps_to_interpolate
        )

        interp_norm_keypoint_sequences.append(
            norm_interpolated_joint_kps
        )

        interp_raw_keypoint_sequences.append(
            raw_interpolated_joint_kps
        )

    player_pose_sequence.keypoints_xy_norm = np.stack(
        interp_norm_keypoint_sequences,
        axis=1
    ).astype(np.float32)

    player_pose_sequence.keypoints_xy_px = np.stack(
        interp_raw_keypoint_sequences,
        axis=1
    ).astype(np.float32)


def _get_missing_keypoint_mask(
    norm_joint_keypoint_sequence: np.ndarray,
    joint_confidence_sequence: np.ndarray,
    detection_missing_mask: np.ndarray,
    min_keypoint_confidence: float
) -> np.ndarray:
    """
    Returns a boolean mask indicating which frames
    have a missing or unreliable keypoint.
    """

    valid_points_mask = (
        ~detection_missing_mask
        & np.all(
            np.isfinite(norm_joint_keypoint_sequence),
            axis=1
        )
        & np.any(
            norm_joint_keypoint_sequence != 0.0,
            axis=1
        )
        & np.isfinite(joint_confidence_sequence)
        & (
            joint_confidence_sequence
            >= min_keypoint_confidence
        )
    )

    return ~valid_points_mask


def _find_interpolatable_gaps(
    missing_points_mask: np.ndarray,
    longest_gap_allowed: int
) -> list[tuple[int, int]]:
    """
    Returns a list of tuples representing
    the indices of the start and end frames
    for all interpolatable gaps for a given
    keypoint.
    """

    num_frames = len(missing_points_mask)

    if num_frames == 0 or not np.any(missing_points_mask):
        return []

    diffs = np.diff(missing_points_mask.astype(np.int8))

    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]

    if missing_points_mask[0]:
        starts = np.r_[0, starts]

    if missing_points_mask[-1]:
        ends = np.r_[ends, num_frames - 1]

    valid_gaps: list[tuple[int, int]] = []

    for start, end in zip(starts, ends):
        gap_length = end - start + 1

        gap_is_short_enough = (
            gap_length <= longest_gap_allowed
        )

        gap_has_left_endpoint = start > 0
        gap_has_right_endpoint = end < num_frames - 1

        if (
            gap_is_short_enough
            and gap_has_left_endpoint
            and gap_has_right_endpoint
        ):
            valid_gaps.append((int(start), int(end)))

    return valid_gaps


def _interpolate_gaps(
    joint_keypoint_sequence: np.ndarray,
    missing_points_mask: np.ndarray,
    gaps_to_interpolate: list[tuple[int, int]]
) -> np.ndarray:
    """
    Returns the interpolated version of the
    x,y point sequence for a given joint's
    keypoint sequence.

    Supports both raw and normalized keypoint
    sequences.
    """

    joint_keypoint_sequence_copy = np.asarray(
        joint_keypoint_sequence,
        dtype=np.float32
    ).copy()

    joint_keypoint_sequence_copy[missing_points_mask] = np.nan

    for start, end in gaps_to_interpolate:
        left_point = joint_keypoint_sequence_copy[start - 1]
        right_point = joint_keypoint_sequence_copy[end + 1]

        if (
            not np.all(np.isfinite(left_point))
            or not np.all(np.isfinite(right_point))
        ):
            continue

        gap_length = end - start + 1

        for step in range(1, gap_length + 1):
            interpolation_fraction = step / (gap_length + 1)

            interpolated_point = (
                (1.0 - interpolation_fraction) * left_point
                + interpolation_fraction * right_point
            )

            joint_keypoint_sequence_copy[
                start + step - 1
            ] = interpolated_point

    return joint_keypoint_sequence_copy
