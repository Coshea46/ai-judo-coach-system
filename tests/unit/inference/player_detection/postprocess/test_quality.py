import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
)
from ai_judo_coach.inference.player_detection.postprocess.quality import (
    PoseSequenceQualityReport,
    assess_pose_sequence_quality,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


KEYPOINT_COUNT = 17


def _create_player_pose_sequence(
    frame_count: int = 10,
) -> PlayerPoseSequence:
    """Create a fully resolved player pose sequence."""

    return PlayerPoseSequence(
        keypoints_xy_px=np.full(
            (frame_count, KEYPOINT_COUNT, 2),
            100.0,
            dtype=np.float32,
        ),
        keypoints_xy_norm=np.full(
            (frame_count, KEYPOINT_COUNT, 2),
            0.5,
            dtype=np.float32,
        ),
        keypoints_conf=np.full(
            (frame_count, KEYPOINT_COUNT),
            0.9,
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
    frame_count: int = 10,
) -> TwoPlayerPoseSequences:
    """Create fully resolved pose sequences for both players."""

    return TwoPlayerPoseSequences(
        clip_id="clip_0",
        player_a_pose_sequence=(
            _create_player_pose_sequence(
                frame_count=frame_count,
            )
        ),
        player_b_pose_sequence=(
            _create_player_pose_sequence(
                frame_count=frame_count,
            )
        ),
    )


def _make_frames_unusable(
    player_pose_sequence: PlayerPoseSequence,
    frame_indices: list[int],
) -> None:
    """Make all normalised keypoints non-finite in selected frames."""

    player_pose_sequence.keypoints_xy_norm[
        frame_indices
    ] = np.nan


def test_assess_pose_sequence_quality_accepts_fully_resolved_sequences() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    assert result == PoseSequenceQualityReport(
        accepted=True,
        rejection_reasons=(),
        player_a_unusable_frame_fraction=0.0,
        player_b_unusable_frame_fraction=0.0,
        player_a_longest_unusable_gap=0,
        player_b_longest_unusable_gap=0,
        both_players_unusable_fraction=0.0,
    )


def test_assess_pose_sequence_quality_reports_all_metrics() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    _make_frames_unusable(
        player_pose_sequence=(
            pose_sequences.player_a_pose_sequence
        ),
        frame_indices=[1, 2, 7],
    )
    _make_frames_unusable(
        player_pose_sequence=(
            pose_sequences.player_b_pose_sequence
        ),
        frame_indices=[2, 5, 6, 7],
    )

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            max_unusable_frame_fraction_per_player=0.5,
            max_consecutive_unusable_frames_per_player=5,
        ),
    )

    assert result.accepted is True
    assert result.rejection_reasons == ()

    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(0.3)
    )
    assert (
        result.player_b_unusable_frame_fraction
        == pytest.approx(0.4)
    )

    assert result.player_a_longest_unusable_gap == 2
    assert result.player_b_longest_unusable_gap == 3

    # Both players are unusable in frames 2 and 7.
    assert (
        result.both_players_unusable_fraction
        == pytest.approx(0.2)
    )


def test_assess_pose_sequence_quality_accepts_values_at_thresholds() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    _make_frames_unusable(
        player_pose_sequence=(
            pose_sequences.player_a_pose_sequence
        ),
        frame_indices=[3, 4],
    )
    _make_frames_unusable(
        player_pose_sequence=(
            pose_sequences.player_b_pose_sequence
        ),
        frame_indices=[7, 8],
    )

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            max_unusable_frame_fraction_per_player=0.2,
            max_consecutive_unusable_frames_per_player=2,
        ),
    )

    assert result.accepted is True
    assert result.rejection_reasons == ()
    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(0.2)
    )
    assert (
        result.player_b_unusable_frame_fraction
        == pytest.approx(0.2)
    )
    assert result.player_a_longest_unusable_gap == 2
    assert result.player_b_longest_unusable_gap == 2


@pytest.mark.parametrize(
    (
        "player_name",
        "expected_reason",
    ),
    [
        (
            "player_a",
            "player_a_unusable_frame_fraction_exceeded",
        ),
        (
            "player_b",
            "player_b_unusable_frame_fraction_exceeded",
        ),
    ],
)
def test_assess_pose_sequence_quality_rejects_excessive_unusable_fraction(
    player_name: str,
    expected_reason: str,
) -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    player_sequence = (
        pose_sequences.player_a_pose_sequence
        if player_name == "player_a"
        else pose_sequences.player_b_pose_sequence
    )

    # The separated frames keep the longest gap at one.
    _make_frames_unusable(
        player_pose_sequence=player_sequence,
        frame_indices=[1, 3, 5],
    )

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            max_unusable_frame_fraction_per_player=0.2,
            max_consecutive_unusable_frames_per_player=2,
        ),
    )

    assert result.accepted is False
    assert result.rejection_reasons == (
        expected_reason,
    )


@pytest.mark.parametrize(
    (
        "player_name",
        "expected_reason",
    ),
    [
        (
            "player_a",
            "player_a_max_consecutive_unusable_frames_exceeded",
        ),
        (
            "player_b",
            "player_b_max_consecutive_unusable_frames_exceeded",
        ),
    ],
)
def test_assess_pose_sequence_quality_rejects_excessive_consecutive_gap(
    player_name: str,
    expected_reason: str,
) -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=20,
        )
    )

    player_sequence = (
        pose_sequences.player_a_pose_sequence
        if player_name == "player_a"
        else pose_sequences.player_b_pose_sequence
    )

    # Three unusable frames are 15% of the sequence, so only the
    # consecutive-gap threshold is exceeded.
    _make_frames_unusable(
        player_pose_sequence=player_sequence,
        frame_indices=[5, 6, 7],
    )

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            max_unusable_frame_fraction_per_player=0.2,
            max_consecutive_unusable_frames_per_player=2,
        ),
    )

    assert result.accepted is False
    assert result.rejection_reasons == (
        expected_reason,
    )


def test_assess_pose_sequence_quality_reports_all_rejection_reasons() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    _make_frames_unusable(
        player_pose_sequence=(
            pose_sequences.player_a_pose_sequence
        ),
        frame_indices=[0, 1, 2],
    )
    _make_frames_unusable(
        player_pose_sequence=(
            pose_sequences.player_b_pose_sequence
        ),
        frame_indices=[5, 6, 7, 8],
    )

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            max_unusable_frame_fraction_per_player=0.2,
            max_consecutive_unusable_frames_per_player=2,
        ),
    )

    assert result.accepted is False
    assert result.rejection_reasons == (
        "player_a_unusable_frame_fraction_exceeded",
        "player_b_unusable_frame_fraction_exceeded",
        "player_a_max_consecutive_unusable_frames_exceeded",
        "player_b_max_consecutive_unusable_frames_exceeded",
    )

    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(0.3)
    )
    assert (
        result.player_b_unusable_frame_fraction
        == pytest.approx(0.4)
    )
    assert result.player_a_longest_unusable_gap == 3
    assert result.player_b_longest_unusable_gap == 4
    assert (
        result.both_players_unusable_fraction
        == pytest.approx(0.0)
    )


def test_frame_is_usable_with_exact_minimum_resolved_keypoints() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=2,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[0] = np.nan
    player_a.keypoints_xy_norm[
        0,
        :6,
    ] = 0.5

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            min_resolved_keypoints_per_usable_frame=6,
            max_unusable_frame_fraction_per_player=1.0,
            max_consecutive_unusable_frames_per_player=10,
        ),
    )

    assert result.accepted is True
    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(0.0)
    )
    assert result.player_a_longest_unusable_gap == 0


def test_frame_is_unusable_below_minimum_resolved_keypoints() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=2,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[0] = np.nan
    player_a.keypoints_xy_norm[
        0,
        :5,
    ] = 0.5

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            min_resolved_keypoints_per_usable_frame=6,
            max_unusable_frame_fraction_per_player=1.0,
            max_consecutive_unusable_frames_per_player=10,
        ),
    )

    assert result.accepted is True
    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(0.5)
    )
    assert result.player_a_longest_unusable_gap == 1


@pytest.mark.parametrize(
    "non_finite_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_keypoint_is_unresolved_if_either_coordinate_is_non_finite(
    non_finite_value: float,
) -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=1,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[
        0,
        0,
        0,
    ] = non_finite_value

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            min_resolved_keypoints_per_usable_frame=17,
            max_unusable_frame_fraction_per_player=1.0,
            max_consecutive_unusable_frames_per_player=10,
        ),
    )

    assert result.accepted is True
    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(1.0)
    )
    assert result.player_a_longest_unusable_gap == 1


def test_quality_assessment_uses_only_normalized_coordinate_finiteness() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=3,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[:] = 0.0
    player_a.keypoints_conf[:] = np.nan
    player_a.missing_mask[:] = True

    result = assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    # Quality assessment is performed after interpolation. At this
    # boundary, any finite coordinate pair is considered resolved.
    assert result.accepted is True
    assert (
        result.player_a_unusable_frame_fraction
        == pytest.approx(0.0)
    )
    assert result.player_a_longest_unusable_gap == 0


def test_assess_pose_sequence_quality_does_not_modify_pose_sequences() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=5,
        )
    )

    player_a_before = (
        pose_sequences
        .player_a_pose_sequence
        .keypoints_xy_norm
        .copy()
    )
    player_b_before = (
        pose_sequences
        .player_b_pose_sequence
        .keypoints_xy_norm
        .copy()
    )

    assess_pose_sequence_quality(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    np.testing.assert_array_equal(
        pose_sequences
        .player_a_pose_sequence
        .keypoints_xy_norm,
        player_a_before,
    )
    np.testing.assert_array_equal(
        pose_sequences
        .player_b_pose_sequence
        .keypoints_xy_norm,
        player_b_before,
    )


@pytest.mark.parametrize(
    "malformed_shape",
    [
        (10, 16, 2),
        (10, 17, 3),
        (10, 34),
        (1, 10, 17, 2),
    ],
)
def test_assess_pose_sequence_quality_rejects_malformed_player_a_shape(
    malformed_shape: tuple[int, ...],
) -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    pose_sequences.player_a_pose_sequence.keypoints_xy_norm = (
        np.zeros(
            malformed_shape,
            dtype=np.float32,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "player_a keypoints_xy_norm must have "
            r"shape \[T, 17, 2\]"
        ),
    ):
        assess_pose_sequence_quality(
            clip_player_pose_sequences=pose_sequences,
            config=PlayerDetectionConfig(),
        )


@pytest.mark.parametrize(
    "malformed_shape",
    [
        (10, 16, 2),
        (10, 17, 3),
        (10, 34),
        (1, 10, 17, 2),
    ],
)
def test_assess_pose_sequence_quality_rejects_malformed_player_b_shape(
    malformed_shape: tuple[int, ...],
) -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=10,
        )
    )

    pose_sequences.player_b_pose_sequence.keypoints_xy_norm = (
        np.zeros(
            malformed_shape,
            dtype=np.float32,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "player_b keypoints_xy_norm must have "
            r"shape \[T, 17, 2\]"
        ),
    ):
        assess_pose_sequence_quality(
            clip_player_pose_sequences=pose_sequences,
            config=PlayerDetectionConfig(),
        )


def test_assess_pose_sequence_quality_rejects_different_frame_counts() -> None:
    pose_sequences = TwoPlayerPoseSequences(
        clip_id="clip_0",
        player_a_pose_sequence=(
            _create_player_pose_sequence(
                frame_count=10,
            )
        ),
        player_b_pose_sequence=(
            _create_player_pose_sequence(
                frame_count=9,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Player A and player B pose sequences must "
            "contain the same number of frames"
        ),
    ):
        assess_pose_sequence_quality(
            clip_player_pose_sequences=pose_sequences,
            config=PlayerDetectionConfig(),
        )


def test_assess_pose_sequence_quality_rejects_empty_sequences() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=0,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Pose sequences must contain at least one frame"
        ),
    ):
        assess_pose_sequence_quality(
            clip_player_pose_sequences=pose_sequences,
            config=PlayerDetectionConfig(),
        )
