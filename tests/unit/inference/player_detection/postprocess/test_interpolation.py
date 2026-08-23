import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
)
from ai_judo_coach.inference.player_detection.postprocess.interpolation import (
    interpolate_two_player_pose_sequences_in_place,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


KEYPOINT_COUNT = 17


def _create_player_pose_sequence(
    frame_count: int = 7,
    coordinate_offset: float = 0.0,
    dtype: np.dtype = np.dtype(np.float32),
) -> PlayerPoseSequence:
    """Create a complete, linearly changing player pose sequence."""

    keypoints_xy_norm = np.empty(
        (frame_count, KEYPOINT_COUNT, 2),
        dtype=dtype,
    )

    for frame_idx in range(frame_count):
        for keypoint_idx in range(KEYPOINT_COUNT):
            keypoints_xy_norm[
                frame_idx,
                keypoint_idx,
            ] = [
                (
                    0.10
                    + coordinate_offset
                    + (frame_idx * 0.05)
                    + (keypoint_idx * 0.001)
                ),
                (
                    0.20
                    + coordinate_offset
                    + (frame_idx * 0.04)
                    + (keypoint_idx * 0.001)
                ),
            ]

    keypoints_xy_px = (
        keypoints_xy_norm * 1000.0
    ).astype(dtype)

    return PlayerPoseSequence(
        keypoints_xy_px=keypoints_xy_px,
        keypoints_xy_norm=keypoints_xy_norm,
        keypoints_conf=np.full(
            (frame_count, KEYPOINT_COUNT),
            0.9,
            dtype=np.float32,
        ),
        missing_mask=np.zeros(
            frame_count,
            dtype=bool,
        ),
        source_detection_idx=np.arange(
            frame_count,
            dtype=np.int32,
        ),
        source_track_id=np.full(
            frame_count,
            10,
            dtype=np.int32,
        ),
    )


def _create_two_player_pose_sequences(
    frame_count: int = 7,
) -> TwoPlayerPoseSequences:
    """Create complete pose sequences for two players."""

    return TwoPlayerPoseSequences(
        clip_id="clip_0",
        player_a_pose_sequence=(
            _create_player_pose_sequence(
                frame_count=frame_count,
                coordinate_offset=0.0,
            )
        ),
        player_b_pose_sequence=(
            _create_player_pose_sequence(
                frame_count=frame_count,
                coordinate_offset=0.25,
            )
        ),
    )


def test_interpolate_two_player_pose_sequences_modifies_both_players_in_place() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )

    player_a = pose_sequences.player_a_pose_sequence
    player_b = pose_sequences.player_b_pose_sequence

    expected_player_a_norm = (
        player_a.keypoints_xy_norm.copy()
    )
    expected_player_a_px = (
        player_a.keypoints_xy_px.copy()
    )
    expected_player_b_norm = (
        player_b.keypoints_xy_norm.copy()
    )
    expected_player_b_px = (
        player_b.keypoints_xy_px.copy()
    )

    player_a.keypoints_xy_norm[2, 0] = [0.0, 0.0]
    player_a.keypoints_xy_px[2, 0] = [-1.0, -1.0]
    player_a.keypoints_conf[2, 0] = 0.1

    # A missing detection makes every keypoint in the frame missing.
    player_b.missing_mask[3] = True
    player_b.keypoints_xy_norm[3] = 0.0
    player_b.keypoints_xy_px[3] = -1.0

    result = interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=5,
        ),
    )

    assert result is None

    np.testing.assert_allclose(
        player_a.keypoints_xy_norm,
        expected_player_a_norm,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        player_a.keypoints_xy_px,
        expected_player_a_px,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        player_b.keypoints_xy_norm,
        expected_player_b_norm,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        player_b.keypoints_xy_px,
        expected_player_b_px,
        atol=1e-4,
    )

    assert player_a.keypoints_conf[2, 0] == pytest.approx(
        0.1
    )
    assert player_b.missing_mask[3]


def test_interpolation_is_linear_across_multi_frame_gap() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    keypoint_idx = 0

    player_a.keypoints_xy_norm[1, keypoint_idx] = [
        0.2,
        0.4,
    ]
    player_a.keypoints_xy_norm[5, keypoint_idx] = [
        0.8,
        1.0,
    ]

    player_a.keypoints_xy_px[1, keypoint_idx] = [
        20.0,
        40.0,
    ]
    player_a.keypoints_xy_px[5, keypoint_idx] = [
        80.0,
        100.0,
    ]

    player_a.keypoints_xy_norm[
        2:5,
        keypoint_idx,
    ] = 0.0
    player_a.keypoints_xy_px[
        2:5,
        keypoint_idx,
    ] = -1.0
    player_a.keypoints_conf[
        2:5,
        keypoint_idx,
    ] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=3,
        ),
    )

    np.testing.assert_allclose(
        player_a.keypoints_xy_norm[
            2:5,
            keypoint_idx,
        ],
        np.array(
            [
                [0.35, 0.55],
                [0.50, 0.70],
                [0.65, 0.85],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )

    np.testing.assert_allclose(
        player_a.keypoints_xy_px[
            2:5,
            keypoint_idx,
        ],
        np.array(
            [
                [35.0, 55.0],
                [50.0, 70.0],
                [65.0, 85.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )


def test_interpolation_accepts_gap_at_maximum_allowed_length() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    expected_norm = (
        player_a.keypoints_xy_norm.copy()
    )
    expected_px = (
        player_a.keypoints_xy_px.copy()
    )

    player_a.keypoints_xy_norm[2:5, 4] = 0.0
    player_a.keypoints_xy_px[2:5, 4] = 0.0
    player_a.keypoints_conf[2:5, 4] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=3,
        ),
    )

    np.testing.assert_allclose(
        player_a.keypoints_xy_norm[:, 4],
        expected_norm[:, 4],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        player_a.keypoints_xy_px[:, 4],
        expected_px[:, 4],
        atol=1e-4,
    )


def test_gap_longer_than_allowed_is_left_unresolved() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=8,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[2:6, 5] = 0.0
    player_a.keypoints_xy_px[2:6, 5] = -1.0
    player_a.keypoints_conf[2:6, 5] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=3,
        ),
    )

    assert np.all(
        np.isnan(
            player_a.keypoints_xy_norm[2:6, 5]
        )
    )
    assert np.all(
        np.isnan(
            player_a.keypoints_xy_px[2:6, 5]
        )
    )


def test_leading_gap_is_not_interpolated() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[0:2, 2] = 0.0
    player_a.keypoints_xy_px[0:2, 2] = -1.0
    player_a.keypoints_conf[0:2, 2] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=5,
        ),
    )

    assert np.all(
        np.isnan(
            player_a.keypoints_xy_norm[0:2, 2]
        )
    )
    assert np.all(
        np.isnan(
            player_a.keypoints_xy_px[0:2, 2]
        )
    )


def test_trailing_gap_is_not_interpolated() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[5:7, 2] = 0.0
    player_a.keypoints_xy_px[5:7, 2] = -1.0
    player_a.keypoints_conf[5:7, 2] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=5,
        ),
    )

    assert np.all(
        np.isnan(
            player_a.keypoints_xy_norm[5:7, 2]
        )
    )
    assert np.all(
        np.isnan(
            player_a.keypoints_xy_px[5:7, 2]
        )
    )


def test_interpolates_eligible_gap_but_not_long_gap_in_same_sequence() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=12,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    expected_short_gap_norm = (
        player_a.keypoints_xy_norm[2, 6].copy()
    )
    expected_short_gap_px = (
        player_a.keypoints_xy_px[2, 6].copy()
    )

    player_a.keypoints_xy_norm[2, 6] = 0.0
    player_a.keypoints_xy_px[2, 6] = -1.0
    player_a.keypoints_conf[2, 6] = 0.0

    player_a.keypoints_xy_norm[5:9, 6] = 0.0
    player_a.keypoints_xy_px[5:9, 6] = -1.0
    player_a.keypoints_conf[5:9, 6] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=2,
        ),
    )

    np.testing.assert_allclose(
        player_a.keypoints_xy_norm[2, 6],
        expected_short_gap_norm,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        player_a.keypoints_xy_px[2, 6],
        expected_short_gap_px,
        atol=1e-4,
    )

    assert np.all(
        np.isnan(
            player_a.keypoints_xy_norm[5:9, 6]
        )
    )
    assert np.all(
        np.isnan(
            player_a.keypoints_xy_px[5:9, 6]
        )
    )


@pytest.mark.parametrize(
    "missing_reason",
    [
        "detection_missing",
        "zero_coordinates",
        "nan_coordinate",
        "positive_infinite_coordinate",
        "negative_infinite_coordinate",
        "low_confidence",
        "nan_confidence",
        "infinite_confidence",
    ],
)
def test_interpolation_recognises_each_missing_keypoint_condition(
    missing_reason: str,
) -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=5,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    frame_idx = 2
    keypoint_idx = 8

    expected_norm = (
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
        ].copy()
    )
    expected_px = (
        player_a.keypoints_xy_px[
            frame_idx,
            keypoint_idx,
        ].copy()
    )

    # Make the raw coordinate visibly incorrect so the test confirms
    # that both coordinate arrays are interpolated from their endpoints.
    player_a.keypoints_xy_px[
        frame_idx,
        keypoint_idx,
    ] = [-100.0, -100.0]

    if missing_reason == "detection_missing":
        player_a.missing_mask[frame_idx] = True

    elif missing_reason == "zero_coordinates":
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
        ] = [0.0, 0.0]

    elif missing_reason == "nan_coordinate":
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
            0,
        ] = np.nan

    elif missing_reason == "positive_infinite_coordinate":
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
            0,
        ] = np.inf

    elif missing_reason == "negative_infinite_coordinate":
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
            1,
        ] = -np.inf

    elif missing_reason == "low_confidence":
        player_a.keypoints_conf[
            frame_idx,
            keypoint_idx,
        ] = 0.29

    elif missing_reason == "nan_confidence":
        player_a.keypoints_conf[
            frame_idx,
            keypoint_idx,
        ] = np.nan

    elif missing_reason == "infinite_confidence":
        player_a.keypoints_conf[
            frame_idx,
            keypoint_idx,
        ] = np.inf

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            keypoint_confidence_threshold=0.3,
            longest_gap_allowed=1,
        ),
    )

    np.testing.assert_allclose(
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
        ],
        expected_norm,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        player_a.keypoints_xy_px[
            frame_idx,
            keypoint_idx,
        ],
        expected_px,
        atol=1e-4,
    )


def test_confidence_threshold_is_inclusive() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=5,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    frame_idx = 2
    keypoint_idx = 9

    expected_norm = np.array(
        [0.91, 0.92],
        dtype=np.float32,
    )
    expected_px = np.array(
        [910.0, 920.0],
        dtype=np.float32,
    )

    player_a.keypoints_xy_norm[
        frame_idx,
        keypoint_idx,
    ] = expected_norm
    player_a.keypoints_xy_px[
        frame_idx,
        keypoint_idx,
    ] = expected_px
    player_a.keypoints_conf[
        frame_idx,
        keypoint_idx,
    ] = 0.3

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            keypoint_confidence_threshold=0.3,
        ),
    )

    np.testing.assert_array_equal(
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
        ],
        expected_norm,
    )
    np.testing.assert_array_equal(
        player_a.keypoints_xy_px[
            frame_idx,
            keypoint_idx,
        ],
        expected_px,
    )


def test_keypoint_with_one_non_zero_coordinate_is_valid() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=5,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    frame_idx = 2
    keypoint_idx = 10

    expected_norm = np.array(
        [0.0, 0.75],
        dtype=np.float32,
    )
    expected_px = np.array(
        [0.0, 750.0],
        dtype=np.float32,
    )

    player_a.keypoints_xy_norm[
        frame_idx,
        keypoint_idx,
    ] = expected_norm
    player_a.keypoints_xy_px[
        frame_idx,
        keypoint_idx,
    ] = expected_px
    player_a.keypoints_conf[
        frame_idx,
        keypoint_idx,
    ] = 0.9

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    np.testing.assert_array_equal(
        player_a.keypoints_xy_norm[
            frame_idx,
            keypoint_idx,
        ],
        expected_norm,
    )
    np.testing.assert_array_equal(
        player_a.keypoints_xy_px[
            frame_idx,
            keypoint_idx,
        ],
        expected_px,
    )


def test_interpolation_does_not_modify_sequence_metadata() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    expected_confidence = (
        player_a.keypoints_conf.copy()
    )
    expected_missing_mask = (
        player_a.missing_mask.copy()
    )
    expected_detection_indices = (
        player_a.source_detection_idx.copy()
    )
    expected_track_ids = (
        player_a.source_track_id.copy()
    )

    player_a.keypoints_xy_norm[3, 11] = 0.0
    player_a.keypoints_xy_px[3, 11] = -1.0
    player_a.keypoints_conf[3, 11] = 0.1

    expected_confidence[3, 11] = 0.1

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    np.testing.assert_array_equal(
        player_a.keypoints_conf,
        expected_confidence,
    )
    np.testing.assert_array_equal(
        player_a.missing_mask,
        expected_missing_mask,
    )
    np.testing.assert_array_equal(
        player_a.source_detection_idx,
        expected_detection_indices,
    )
    np.testing.assert_array_equal(
        player_a.source_track_id,
        expected_track_ids,
    )


def test_complete_pose_sequences_keep_their_coordinate_values() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=7,
        )
    )

    player_a = pose_sequences.player_a_pose_sequence
    player_b = pose_sequences.player_b_pose_sequence

    expected_player_a_norm = (
        player_a.keypoints_xy_norm.copy()
    )
    expected_player_a_px = (
        player_a.keypoints_xy_px.copy()
    )
    expected_player_b_norm = (
        player_b.keypoints_xy_norm.copy()
    )
    expected_player_b_px = (
        player_b.keypoints_xy_px.copy()
    )

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    np.testing.assert_array_equal(
        player_a.keypoints_xy_norm,
        expected_player_a_norm,
    )
    np.testing.assert_array_equal(
        player_a.keypoints_xy_px,
        expected_player_a_px,
    )
    np.testing.assert_array_equal(
        player_b.keypoints_xy_norm,
        expected_player_b_norm,
    )
    np.testing.assert_array_equal(
        player_b.keypoints_xy_px,
        expected_player_b_px,
    )


def test_interpolated_coordinate_arrays_are_float32() -> None:
    player_a = _create_player_pose_sequence(
        frame_count=5,
        coordinate_offset=0.0,
        dtype=np.dtype(np.float64),
    )
    player_b = _create_player_pose_sequence(
        frame_count=5,
        coordinate_offset=0.25,
        dtype=np.dtype(np.float64),
    )

    pose_sequences = TwoPlayerPoseSequences(
        clip_id="clip_0",
        player_a_pose_sequence=player_a,
        player_b_pose_sequence=player_b,
    )

    player_a.keypoints_xy_norm[2, 0] = 0.0
    player_a.keypoints_xy_px[2, 0] = 0.0
    player_a.keypoints_conf[2, 0] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(),
    )

    assert player_a.keypoints_xy_norm.dtype == np.float32
    assert player_a.keypoints_xy_px.dtype == np.float32
    assert player_b.keypoints_xy_norm.dtype == np.float32
    assert player_b.keypoints_xy_px.dtype == np.float32

    assert player_a.keypoints_xy_norm.shape == (
        5,
        KEYPOINT_COUNT,
        2,
    )
    assert player_a.keypoints_xy_px.shape == (
        5,
        KEYPOINT_COUNT,
        2,
    )


def test_zero_longest_gap_allowed_leaves_missing_point_unresolved() -> None:
    pose_sequences = (
        _create_two_player_pose_sequences(
            frame_count=5,
        )
    )
    player_a = pose_sequences.player_a_pose_sequence

    player_a.keypoints_xy_norm[2, 0] = 0.0
    player_a.keypoints_xy_px[2, 0] = -1.0
    player_a.keypoints_conf[2, 0] = 0.0

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=pose_sequences,
        config=PlayerDetectionConfig(
            longest_gap_allowed=0,
        ),
    )

    assert np.all(
        np.isnan(
            player_a.keypoints_xy_norm[2, 0]
        )
    )
    assert np.all(
        np.isnan(
            player_a.keypoints_xy_px[2, 0]
        )
    )
