"""
Functions for converting two-player pose sequences
into LSTM input arrays.
"""

import numpy as np

from ai_judo_coach.inference.inference_schemas import TwoPlayerPoseSequences
from ai_judo_coach.inference.player_detection import PoseSequenceQualityReport

from .lstm_input_config import LSTMInputConfig


def build_lstm_input_array(
    clip_player_pose_sequences: TwoPlayerPoseSequences,
    pose_sequence_quality_report: PoseSequenceQualityReport,
) -> np.ndarray | None:
    """
    Convert two-player pose sequences into the normalized
    coordinate array expected by the LSTM.

    Player A occupies columns 0 to 33.
    Player B occupies columns 34 to 67.

    Unresolved non-finite coordinate values are replaced with
    the configured unresolved-coordinate value at this boundary.

    Returns None if the sequence was rejected by quality assessment
    or if either player array has an unexpected shape.

    Otherwise, returns a contiguous array with shape [210, 68]
    and dtype float32.
    """

    if not pose_sequence_quality_report.accepted:
        return None

    config = LSTMInputConfig()

    player_a_keypoints = np.asarray(
        clip_player_pose_sequences
        .player_a_pose_sequence
        .keypoints_xy_norm,
        dtype=np.float32,
    )

    player_b_keypoints = np.asarray(
        clip_player_pose_sequences
        .player_b_pose_sequence
        .keypoints_xy_norm,
        dtype=np.float32,
    )

    if not _has_expected_pose_shape(player_a_keypoints, config):
        return None

    if not _has_expected_pose_shape(player_b_keypoints, config):
        return None

    lstm_input = _join_player_arrays(
        player_a_keypoints=player_a_keypoints,
        player_b_keypoints=player_b_keypoints,
        config=config,
    )

    return _replace_non_finite_values(
        player_keypoint_array=lstm_input,
        config=config,
    )


def _has_expected_pose_shape(
    player_keypoint_array: np.ndarray,
    config: LSTMInputConfig,
) -> bool:
    """Return whether one player's pose array has shape [210, 17, 2]."""

    expected_shape = (
        config.sequence_length,
        config.keypoint_count,
        config.coordinates_per_keypoint,
    )

    return player_keypoint_array.shape == expected_shape


def _join_player_arrays(
    player_a_keypoints: np.ndarray,
    player_b_keypoints: np.ndarray,
    config: LSTMInputConfig,
) -> np.ndarray:
    """
    Flatten and concatenate the player arrays.

    Player A occupies columns 0 to 33.
    Player B occupies columns 34 to 67.
    """

    player_a_flat = player_a_keypoints.reshape(
        config.sequence_length,
        config.features_per_player,
    )

    player_b_flat = player_b_keypoints.reshape(
        config.sequence_length,
        config.features_per_player,
    )

    return np.concatenate(
        (player_a_flat, player_b_flat),
        axis=1,
        dtype=np.float32,
    )


def _replace_non_finite_values(
    player_keypoint_array: np.ndarray,
    config: LSTMInputConfig,
) -> np.ndarray:
    """Replace unresolved non-finite coordinates at the export boundary."""

    replacement = config.unresolved_coordinate_value

    np.nan_to_num(
        player_keypoint_array,
        copy=False,
        nan=replacement,
        posinf=replacement,
        neginf=replacement,
    )

    return np.ascontiguousarray(
        player_keypoint_array,
        dtype=np.float32,
    )
