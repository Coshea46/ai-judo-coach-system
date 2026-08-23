import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
    PersonDetection,
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
)
from ai_judo_coach.inference.player_detection.pose_sequence_builder import (
    build_two_player_pose_sequences,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


def _create_person_detection(
    detection_idx: int,
    track_id: int | None,
    coordinate_value: float,
    confidence_value: float = 0.9,
) -> PersonDetection:
    """Create one person detection for pose-sequence tests."""

    return PersonDetection(
        detection_idx=detection_idx,
        track_id=track_id,
        bbox_xyxy_px=np.array(
            [10.0, 20.0, 50.0, 80.0],
            dtype=np.float32,
        ),
        bbox_xyxy_normalized=np.array(
            [0.1, 0.2, 0.5, 0.8],
            dtype=np.float32,
        ),
        bbox_conf=0.9,
        keypoints_xy_px=np.full(
            (17, 2),
            coordinate_value * 1000.0,
            dtype=np.float32,
        ),
        keypoints_xy_norm=np.full(
            (17, 2),
            coordinate_value,
            dtype=np.float32,
        ),
        keypoints_conf=np.full(
            17,
            confidence_value,
            dtype=np.float32,
        ),
    )


def _create_frame_detections(
    frame_idx: int,
    detections: list[PersonDetection],
) -> FrameDetections:
    """Create one frame containing the supplied detections."""

    return FrameDetections(
        person_detections=detections,
        frame_idx=frame_idx,
        frame_shape_hw=(720, 1280),
    )


def test_build_two_player_pose_sequences_builds_aligned_player_arrays() -> None:
    frame_0_detection_0 = _create_person_detection(
        detection_idx=0,
        track_id=10,
        coordinate_value=0.10,
    )
    frame_0_detection_1 = _create_person_detection(
        detection_idx=1,
        track_id=20,
        coordinate_value=0.20,
    )
    frame_1_detection_0 = _create_person_detection(
        detection_idx=0,
        track_id=11,
        coordinate_value=0.30,
    )
    frame_1_detection_1 = _create_person_detection(
        detection_idx=1,
        track_id=21,
        coordinate_value=0.40,
    )

    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    frame_0_detection_0,
                    frame_0_detection_1,
                ],
            ),
            _create_frame_detections(
                frame_idx=1,
                detections=[
                    frame_1_detection_0,
                    frame_1_detection_1,
                ],
            ),
        ],
        clip_id="clip_42",
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (1, 0),
            (0, 1),
        ],
        config=PlayerDetectionConfig(),
    )

    assert isinstance(
        result,
        TwoPlayerPoseSequences,
    )
    assert result.clip_id == "clip_42"

    assert isinstance(
        result.player_a_pose_sequence,
        PlayerPoseSequence,
    )
    assert isinstance(
        result.player_b_pose_sequence,
        PlayerPoseSequence,
    )

    player_a = result.player_a_pose_sequence
    player_b = result.player_b_pose_sequence

    np.testing.assert_array_equal(
        player_a.keypoints_xy_norm,
        np.stack(
            [
                frame_0_detection_1.keypoints_xy_norm,
                frame_1_detection_0.keypoints_xy_norm,
            ]
        ),
    )
    np.testing.assert_array_equal(
        player_b.keypoints_xy_norm,
        np.stack(
            [
                frame_0_detection_0.keypoints_xy_norm,
                frame_1_detection_1.keypoints_xy_norm,
            ]
        ),
    )

    np.testing.assert_array_equal(
        player_a.keypoints_xy_px,
        np.stack(
            [
                frame_0_detection_1.keypoints_xy_px,
                frame_1_detection_0.keypoints_xy_px,
            ]
        ),
    )
    np.testing.assert_array_equal(
        player_b.keypoints_xy_px,
        np.stack(
            [
                frame_0_detection_0.keypoints_xy_px,
                frame_1_detection_1.keypoints_xy_px,
            ]
        ),
    )

    np.testing.assert_array_equal(
        player_a.keypoints_conf,
        np.stack(
            [
                frame_0_detection_1.keypoints_conf,
                frame_1_detection_0.keypoints_conf,
            ]
        ),
    )
    np.testing.assert_array_equal(
        player_b.keypoints_conf,
        np.stack(
            [
                frame_0_detection_0.keypoints_conf,
                frame_1_detection_1.keypoints_conf,
            ]
        ),
    )

    np.testing.assert_array_equal(
        player_a.source_detection_idx,
        np.array(
            [1, 0],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        player_b.source_detection_idx,
        np.array(
            [0, 1],
            dtype=np.int32,
        ),
    )

    np.testing.assert_array_equal(
        player_a.source_track_id,
        np.array(
            [20, 11],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        player_b.source_track_id,
        np.array(
            [10, 21],
            dtype=np.int32,
        ),
    )

    np.testing.assert_array_equal(
        player_a.missing_mask,
        np.array(
            [False, False],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        player_b.missing_mask,
        np.array(
            [False, False],
            dtype=np.bool_,
        ),
    )


def test_build_two_player_pose_sequences_returns_expected_shapes_and_dtypes() -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    _create_person_detection(
                        detection_idx=0,
                        track_id=10,
                        coordinate_value=0.1,
                    ),
                    _create_person_detection(
                        detection_idx=1,
                        track_id=20,
                        coordinate_value=0.2,
                    ),
                ],
            ),
            _create_frame_detections(
                frame_idx=1,
                detections=[
                    _create_person_detection(
                        detection_idx=0,
                        track_id=10,
                        coordinate_value=0.3,
                    ),
                    _create_person_detection(
                        detection_idx=1,
                        track_id=20,
                        coordinate_value=0.4,
                    ),
                ],
            ),
        ],
        clip_id="clip_0",
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (0, 1),
            (0, 1),
        ],
        config=PlayerDetectionConfig(),
    )

    for player_sequence in (
        result.player_a_pose_sequence,
        result.player_b_pose_sequence,
    ):
        assert player_sequence.keypoints_xy_px.shape == (
            2,
            17,
            2,
        )
        assert player_sequence.keypoints_xy_norm.shape == (
            2,
            17,
            2,
        )
        assert player_sequence.keypoints_conf.shape == (
            2,
            17,
        )
        assert player_sequence.missing_mask.shape == (2,)
        assert player_sequence.source_detection_idx.shape == (2,)
        assert player_sequence.source_track_id.shape == (2,)

        assert (
            player_sequence.keypoints_xy_px.dtype
            == np.float32
        )
        assert (
            player_sequence.keypoints_xy_norm.dtype
            == np.float32
        )
        assert (
            player_sequence.keypoints_conf.dtype
            == np.float32
        )
        assert (
            player_sequence.missing_mask.dtype
            == np.bool_
        )
        assert (
            player_sequence.source_detection_idx.dtype
            == np.int32
        )
        assert (
            player_sequence.source_track_id.dtype
            == np.int32
        )


def test_build_two_player_pose_sequences_pads_missing_player() -> None:
    detected_player = _create_person_detection(
        detection_idx=0,
        track_id=15,
        coordinate_value=0.25,
    )

    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    detected_player,
                ],
            ),
        ],
        clip_id="clip_0",
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (-1, 0),
        ],
        config=PlayerDetectionConfig(),
    )

    missing_player = (
        result.player_a_pose_sequence
    )
    present_player = (
        result.player_b_pose_sequence
    )

    assert missing_player.keypoints_xy_px.shape == (
        1,
        17,
        2,
    )
    assert missing_player.keypoints_xy_norm.shape == (
        1,
        17,
        2,
    )
    assert missing_player.keypoints_conf.shape == (
        1,
        17,
    )

    assert np.all(
        np.isnan(
            missing_player.keypoints_xy_px
        )
    )
    assert np.all(
        np.isnan(
            missing_player.keypoints_xy_norm
        )
    )
    assert np.all(
        missing_player.keypoints_conf == 0.0
    )

    np.testing.assert_array_equal(
        missing_player.missing_mask,
        np.array(
            [True],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        missing_player.source_detection_idx,
        np.array(
            [-1],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        missing_player.source_track_id,
        np.array(
            [-1],
            dtype=np.int32,
        ),
    )

    np.testing.assert_array_equal(
        present_player.keypoints_xy_px[0],
        detected_player.keypoints_xy_px,
    )
    np.testing.assert_array_equal(
        present_player.keypoints_xy_norm[0],
        detected_player.keypoints_xy_norm,
    )
    np.testing.assert_array_equal(
        present_player.keypoints_conf[0],
        detected_player.keypoints_conf,
    )
    np.testing.assert_array_equal(
        present_player.missing_mask,
        np.array(
            [False],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        present_player.source_detection_idx,
        np.array(
            [0],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        present_player.source_track_id,
        np.array(
            [15],
            dtype=np.int32,
        ),
    )


def test_build_two_player_pose_sequences_handles_both_players_missing() -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[],
            ),
        ],
        clip_id="clip_0",
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (-1, -1),
        ],
        config=PlayerDetectionConfig(),
    )

    for player_sequence in (
        result.player_a_pose_sequence,
        result.player_b_pose_sequence,
    ):
        assert np.all(
            np.isnan(
                player_sequence.keypoints_xy_px
            )
        )
        assert np.all(
            np.isnan(
                player_sequence.keypoints_xy_norm
            )
        )
        assert np.all(
            player_sequence.keypoints_conf == 0.0
        )

        np.testing.assert_array_equal(
            player_sequence.missing_mask,
            np.array(
                [True],
                dtype=np.bool_,
            ),
        )
        np.testing.assert_array_equal(
            player_sequence.source_detection_idx,
            np.array(
                [-1],
                dtype=np.int32,
            ),
        )
        np.testing.assert_array_equal(
            player_sequence.source_track_id,
            np.array(
                [-1],
                dtype=np.int32,
            ),
        )


def test_build_two_player_pose_sequences_uses_configured_missing_sentinel() -> None:
    detected_player = _create_person_detection(
        detection_idx=0,
        track_id=12,
        coordinate_value=0.5,
    )

    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    detected_player,
                ],
            ),
        ],
        clip_id="clip_0",
    )

    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (-99, 0),
        ],
        config=config,
    )

    np.testing.assert_array_equal(
        result
        .player_a_pose_sequence
        .source_detection_idx,
        np.array(
            [-99],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        result
        .player_a_pose_sequence
        .source_track_id,
        np.array(
            [-99],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        result
        .player_a_pose_sequence
        .missing_mask,
        np.array(
            [True],
            dtype=np.bool_,
        ),
    )


def test_build_two_player_pose_sequences_replaces_unknown_track_id_with_sentinel() -> None:
    detected_player = _create_person_detection(
        detection_idx=0,
        track_id=None,
        coordinate_value=0.5,
    )

    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    detected_player,
                ],
            ),
        ],
        clip_id="clip_0",
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (0, -99),
        ],
        config=PlayerDetectionConfig(
            missing_detection_sentinel=-99,
        ),
    )

    player_a = result.player_a_pose_sequence

    np.testing.assert_array_equal(
        player_a.source_detection_idx,
        np.array(
            [0],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        player_a.source_track_id,
        np.array(
            [-99],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        player_a.missing_mask,
        np.array(
            [False],
            dtype=np.bool_,
        ),
    )


def test_build_two_player_pose_sequences_marks_zero_confidence_pose_as_missing() -> None:
    zero_confidence_detection = (
        _create_person_detection(
            detection_idx=0,
            track_id=10,
            coordinate_value=0.5,
            confidence_value=0.0,
        )
    )

    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    zero_confidence_detection,
                ],
            ),
        ],
        clip_id="clip_0",
    )

    result = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=[
            (0, -1),
        ],
        config=PlayerDetectionConfig(),
    )

    np.testing.assert_array_equal(
        result
        .player_a_pose_sequence
        .missing_mask,
        np.array(
            [True],
            dtype=np.bool_,
        ),
    )


@pytest.mark.parametrize(
    "frame_player_state_sequence",
    [
        [],
        [
            (0, -1),
            (0, -1),
        ],
    ],
)
def test_build_two_player_pose_sequences_rejects_state_count_mismatch(
    frame_player_state_sequence: list[
        tuple[int, int]
    ],
) -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    _create_person_detection(
                        detection_idx=0,
                        track_id=10,
                        coordinate_value=0.5,
                    ),
                ],
            ),
        ],
        clip_id="clip_0",
    )

    with pytest.raises(
        ValueError,
        match=(
            "frame_player_state_sequence must contain "
            "exactly one state for each frame in "
            "clip_detections"
        ),
    ):
        build_two_player_pose_sequences(
            clip_detections=clip_detections,
            frame_player_state_sequence=(
                frame_player_state_sequence
            ),
            config=PlayerDetectionConfig(),
        )


@pytest.mark.parametrize(
    "invalid_assignment",
    [
        -2,
        1,
        5,
    ],
)
def test_build_two_player_pose_sequences_rejects_invalid_player_a_index(
    invalid_assignment: int,
) -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    _create_person_detection(
                        detection_idx=0,
                        track_id=10,
                        coordinate_value=0.5,
                    ),
                ],
            ),
        ],
        clip_id="clip_0",
    )

    with pytest.raises(
        IndexError,
        match=(
            "Player assignment index is outside the "
            "FrameDetections.person_detections list"
        ),
    ):
        build_two_player_pose_sequences(
            clip_detections=clip_detections,
            frame_player_state_sequence=[
                (
                    invalid_assignment,
                    -1,
                ),
            ],
            config=PlayerDetectionConfig(),
        )


@pytest.mark.parametrize(
    "invalid_assignment",
    [
        -2,
        1,
        5,
    ],
)
def test_build_two_player_pose_sequences_rejects_invalid_player_b_index(
    invalid_assignment: int,
) -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                frame_idx=0,
                detections=[
                    _create_person_detection(
                        detection_idx=0,
                        track_id=10,
                        coordinate_value=0.5,
                    ),
                ],
            ),
        ],
        clip_id="clip_0",
    )

    with pytest.raises(
        IndexError,
        match=(
            "Player assignment index is outside the "
            "FrameDetections.person_detections list"
        ),
    ):
        build_two_player_pose_sequences(
            clip_detections=clip_detections,
            frame_player_state_sequence=[
                (
                    0,
                    invalid_assignment,
                ),
            ],
            config=PlayerDetectionConfig(),
        )
