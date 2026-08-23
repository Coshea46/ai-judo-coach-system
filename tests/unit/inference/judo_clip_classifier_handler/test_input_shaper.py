import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
)
from ai_judo_coach.inference.judo_clip_classifier_handler.input_shaper import (
    build_lstm_input_array,
)
from ai_judo_coach.inference.player_detection import (
    PoseSequenceQualityReport,
)


SEQUENCE_LENGTH = 210
KEYPOINT_COUNT = 17
COORDINATES_PER_KEYPOINT = 2
FEATURES_PER_PLAYER = 34
TOTAL_FEATURE_COUNT = 68


def _create_player_pose_sequence(
    keypoints_xy_norm: np.ndarray,
) -> PlayerPoseSequence:
    """Create one player pose sequence for input-shaper tests."""

    frame_count = keypoints_xy_norm.shape[0]

    return PlayerPoseSequence(
        keypoints_xy_px=np.zeros(
            (frame_count, KEYPOINT_COUNT, COORDINATES_PER_KEYPOINT),
            dtype=np.float32,
        ),
        keypoints_xy_norm=keypoints_xy_norm,
        keypoints_conf=np.ones(
            (frame_count, KEYPOINT_COUNT),
            dtype=np.float32,
        ),
        missing_mask=np.zeros(
            frame_count,
            dtype=bool,
        ),
        source_detection_idx=np.zeros(
            frame_count,
            dtype=np.int32,
        ),
        source_track_id=np.zeros(
            frame_count,
            dtype=np.int32,
        ),
    )


def _create_two_player_pose_sequences(
    player_a_keypoints: np.ndarray,
    player_b_keypoints: np.ndarray,
) -> TwoPlayerPoseSequences:
    """Create two-player pose sequences for testing."""

    return TwoPlayerPoseSequences(
        clip_id="clip_0",
        player_a_pose_sequence=(
            _create_player_pose_sequence(
                keypoints_xy_norm=player_a_keypoints,
            )
        ),
        player_b_pose_sequence=(
            _create_player_pose_sequence(
                keypoints_xy_norm=player_b_keypoints,
            )
        ),
    )


def _create_quality_report(
    accepted: bool = True,
) -> PoseSequenceQualityReport:
    """Create a pose-sequence quality report for testing."""

    return PoseSequenceQualityReport(
        accepted=accepted,
        rejection_reasons=(
            ()
            if accepted
            else ("test_rejection",)
        ),
        player_a_unusable_frame_fraction=0.0,
        player_b_unusable_frame_fraction=0.0,
        player_a_longest_unusable_gap=0,
        player_b_longest_unusable_gap=0,
        both_players_unusable_fraction=0.0,
    )


def test_build_lstm_input_array_returns_none_when_quality_rejected() -> None:
    player_a_keypoints = np.zeros(
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        ),
        dtype=np.float32,
    )
    player_b_keypoints = np.zeros_like(
        player_a_keypoints
    )

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report(
                accepted=False,
            )
        ),
    )

    assert result is None


def test_build_lstm_input_array_flattens_and_joins_player_arrays() -> None:
    player_a_keypoints = np.arange(
        SEQUENCE_LENGTH
        * KEYPOINT_COUNT
        * COORDINATES_PER_KEYPOINT,
        dtype=np.float64,
    ).reshape(
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        )
    )

    player_b_keypoints = (
        player_a_keypoints
        + 100_000.0
    )

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report()
        ),
    )

    assert result is not None
    assert result.shape == (
        SEQUENCE_LENGTH,
        TOTAL_FEATURE_COUNT,
    )
    assert result.dtype == np.float32
    assert result.flags.c_contiguous

    expected_player_a = player_a_keypoints.astype(
        np.float32
    ).reshape(
        SEQUENCE_LENGTH,
        FEATURES_PER_PLAYER,
    )

    expected_player_b = player_b_keypoints.astype(
        np.float32
    ).reshape(
        SEQUENCE_LENGTH,
        FEATURES_PER_PLAYER,
    )

    np.testing.assert_array_equal(
        result[:, :FEATURES_PER_PLAYER],
        expected_player_a,
    )
    np.testing.assert_array_equal(
        result[:, FEATURES_PER_PLAYER:],
        expected_player_b,
    )


def test_build_lstm_input_array_places_players_in_expected_columns() -> None:
    player_a_keypoints = np.zeros(
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        ),
        dtype=np.float32,
    )
    player_b_keypoints = np.zeros_like(
        player_a_keypoints
    )

    player_a_keypoints[0, 0] = [0.1, 0.2]
    player_a_keypoints[0, 1] = [0.3, 0.4]

    player_b_keypoints[0, 0] = [0.5, 0.6]
    player_b_keypoints[0, 1] = [0.7, 0.8]

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report()
        ),
    )

    assert result is not None

    np.testing.assert_allclose(
        result[0, 0:4],
        np.array(
            [0.1, 0.2, 0.3, 0.4],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        result[0, 34:38],
        np.array(
            [0.5, 0.6, 0.7, 0.8],
            dtype=np.float32,
        ),
    )


def test_build_lstm_input_array_replaces_non_finite_values() -> None:
    player_a_keypoints = np.full(
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        ),
        0.25,
        dtype=np.float32,
    )
    player_b_keypoints = np.full_like(
        player_a_keypoints,
        0.75,
    )

    player_a_keypoints[0, 0, 0] = np.nan
    player_a_keypoints[1, 1, 1] = np.inf
    player_b_keypoints[2, 2, 0] = -np.inf

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report()
        ),
    )

    assert result is not None
    assert np.all(np.isfinite(result))

    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 3] == pytest.approx(0.0)
    assert result[2, 38] == pytest.approx(0.0)

    assert result[0, 1] == pytest.approx(0.25)
    assert result[0, 34] == pytest.approx(0.75)


@pytest.mark.parametrize(
    (
        "malformed_player",
        "malformed_shape",
    ),
    [
        (
            "player_a",
            (
                SEQUENCE_LENGTH - 1,
                KEYPOINT_COUNT,
                COORDINATES_PER_KEYPOINT,
            ),
        ),
        (
            "player_a",
            (
                SEQUENCE_LENGTH,
                KEYPOINT_COUNT - 1,
                COORDINATES_PER_KEYPOINT,
            ),
        ),
        (
            "player_a",
            (
                SEQUENCE_LENGTH,
                KEYPOINT_COUNT,
                COORDINATES_PER_KEYPOINT + 1,
            ),
        ),
        (
            "player_b",
            (
                SEQUENCE_LENGTH - 1,
                KEYPOINT_COUNT,
                COORDINATES_PER_KEYPOINT,
            ),
        ),
        (
            "player_b",
            (
                SEQUENCE_LENGTH,
                KEYPOINT_COUNT - 1,
                COORDINATES_PER_KEYPOINT,
            ),
        ),
        (
            "player_b",
            (
                SEQUENCE_LENGTH,
                KEYPOINT_COUNT,
                COORDINATES_PER_KEYPOINT + 1,
            ),
        ),
    ],
)
def test_build_lstm_input_array_returns_none_for_unexpected_pose_shape(
    malformed_player: str,
    malformed_shape: tuple[int, ...],
) -> None:
    expected_shape = (
        SEQUENCE_LENGTH,
        KEYPOINT_COUNT,
        COORDINATES_PER_KEYPOINT,
    )

    player_a_shape = (
        malformed_shape
        if malformed_player == "player_a"
        else expected_shape
    )
    player_b_shape = (
        malformed_shape
        if malformed_player == "player_b"
        else expected_shape
    )

    player_a_keypoints = np.zeros(
        player_a_shape,
        dtype=np.float32,
    )
    player_b_keypoints = np.zeros(
        player_b_shape,
        dtype=np.float32,
    )

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report()
        ),
    )

    assert result is None


@pytest.mark.parametrize(
    "malformed_shape",
    [
        (SEQUENCE_LENGTH, TOTAL_FEATURE_COUNT),
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
        ),
        (
            1,
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        ),
    ],
)
def test_build_lstm_input_array_rejects_wrong_number_of_dimensions(
    malformed_shape: tuple[int, ...],
) -> None:
    player_a_keypoints = np.zeros(
        malformed_shape,
        dtype=np.float32,
    )
    player_b_keypoints = np.zeros(
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        ),
        dtype=np.float32,
    )

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report()
        ),
    )

    assert result is None


def test_build_lstm_input_array_returns_contiguous_output_for_non_contiguous_inputs() -> None:
    player_a_base = np.arange(
        SEQUENCE_LENGTH
        * KEYPOINT_COUNT
        * COORDINATES_PER_KEYPOINT,
        dtype=np.float32,
    ).reshape(
        (
            SEQUENCE_LENGTH,
            KEYPOINT_COUNT,
            COORDINATES_PER_KEYPOINT,
        )
    )
    player_b_base = player_a_base + 1_000.0

    player_a_keypoints = player_a_base[:, ::-1, :]
    player_b_keypoints = player_b_base[:, ::-1, :]

    assert not player_a_keypoints.flags.c_contiguous
    assert not player_b_keypoints.flags.c_contiguous

    result = build_lstm_input_array(
        clip_player_pose_sequences=(
            _create_two_player_pose_sequences(
                player_a_keypoints=player_a_keypoints,
                player_b_keypoints=player_b_keypoints,
            )
        ),
        pose_sequence_quality_report=(
            _create_quality_report()
        ),
    )

    assert result is not None
    assert result.flags.c_contiguous
    assert result.dtype == np.float32
    assert result.shape == (
        SEQUENCE_LENGTH,
        TOTAL_FEATURE_COUNT,
    )

    np.testing.assert_array_equal(
        result[:, :FEATURES_PER_PLAYER],
        player_a_keypoints.reshape(
            SEQUENCE_LENGTH,
            FEATURES_PER_PLAYER,
        ),
    )
    np.testing.assert_array_equal(
        result[:, FEATURES_PER_PLAYER:],
        player_b_keypoints.reshape(
            SEQUENCE_LENGTH,
            FEATURES_PER_PLAYER,
        ),
    )
