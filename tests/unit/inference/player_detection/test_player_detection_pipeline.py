from unittest.mock import call

import pytest

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
)
from ai_judo_coach.inference.player_detection.player_detection_pipeline import (
    detect_players,
)
from ai_judo_coach.inference.player_detection.postprocess import (
    PoseSequenceQualityReport,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


PLAYER_DETECTION_PIPELINE_MODULE_PATH = (
    "ai_judo_coach.inference.player_detection."
    "player_detection_pipeline"
)


def _create_clip_detections() -> ClipDetections:
    """Create a minimal clip for player-detection pipeline tests."""

    return ClipDetections(
        frame_detections=[
            FrameDetections(
                person_detections=[],
                frame_idx=0,
                frame_shape_hw=(720, 1280),
            ),
        ],
        clip_id="clip_0",
    )


def _create_quality_report() -> PoseSequenceQualityReport:
    """Create an accepted pose-sequence quality report."""

    return PoseSequenceQualityReport(
        accepted=True,
        rejection_reasons=(),
        player_a_unusable_frame_fraction=0.0,
        player_b_unusable_frame_fraction=0.0,
        player_a_longest_unusable_gap=0,
        player_b_longest_unusable_gap=0,
        both_players_unusable_fraction=0.0,
    )


def test_detect_players_runs_complete_player_detection_pipeline(
    mocker,
) -> None:
    clip_detections = _create_clip_detections()

    expected_config = PlayerDetectionConfig()
    expected_candidate_states = [
        [
            (-1, -1),
        ],
    ]
    expected_state_sequence = [
        (-1, -1),
    ]
    expected_pose_sequences = mocker.Mock()
    expected_quality_report = (
        _create_quality_report()
    )

    config_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "PlayerDetectionConfig",
        return_value=expected_config,
    )
    generate_states_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "generate_candidate_states_for_clip",
        return_value=expected_candidate_states,
    )
    viterbi_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "viterbi_algorithm",
        return_value=expected_state_sequence,
    )
    build_sequences_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "build_two_player_pose_sequences",
        return_value=expected_pose_sequences,
    )
    interpolation_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "interpolate_two_player_pose_sequences_in_place",
    )
    quality_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "assess_pose_sequence_quality",
        return_value=expected_quality_report,
    )

    result = detect_players(
        clip_detections=clip_detections,
    )

    assert result == (
        expected_pose_sequences,
        expected_quality_report,
    )
    assert result[0] is expected_pose_sequences
    assert result[1] is expected_quality_report

    config_mock.assert_called_once_with()

    generate_states_mock.assert_called_once_with(
        clip_detections=clip_detections,
        config=expected_config,
    )

    viterbi_mock.assert_called_once_with(
        candidate_states_by_frame=(
            expected_candidate_states
        ),
        clip_detections=clip_detections,
        config=expected_config,
    )

    build_sequences_mock.assert_called_once_with(
        clip_detections=clip_detections,
        frame_player_state_sequence=(
            expected_state_sequence
        ),
        config=expected_config,
    )

    interpolation_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            expected_pose_sequences
        ),
        config=expected_config,
    )

    quality_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            expected_pose_sequences
        ),
        config=expected_config,
    )


def test_detect_players_performs_steps_in_expected_order(
    mocker,
) -> None:
    clip_detections = _create_clip_detections()

    expected_config = PlayerDetectionConfig()
    expected_candidate_states = [
        [
            (-1, -1),
        ],
    ]
    expected_state_sequence = [
        (-1, -1),
    ]
    expected_pose_sequences = mocker.Mock()
    expected_quality_report = (
        _create_quality_report()
    )

    call_order = mocker.Mock()

    config_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "PlayerDetectionConfig",
        return_value=expected_config,
    )
    generate_states_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "generate_candidate_states_for_clip",
        return_value=expected_candidate_states,
    )
    viterbi_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "viterbi_algorithm",
        return_value=expected_state_sequence,
    )
    build_sequences_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "build_two_player_pose_sequences",
        return_value=expected_pose_sequences,
    )
    interpolation_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "interpolate_two_player_pose_sequences_in_place",
    )
    quality_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "assess_pose_sequence_quality",
        return_value=expected_quality_report,
    )

    call_order.attach_mock(
        config_mock,
        "create_config",
    )
    call_order.attach_mock(
        generate_states_mock,
        "generate_states",
    )
    call_order.attach_mock(
        viterbi_mock,
        "run_viterbi",
    )
    call_order.attach_mock(
        build_sequences_mock,
        "build_sequences",
    )
    call_order.attach_mock(
        interpolation_mock,
        "interpolate",
    )
    call_order.attach_mock(
        quality_mock,
        "assess_quality",
    )

    detect_players(
        clip_detections=clip_detections,
    )

    assert call_order.mock_calls == [
        call.create_config(),
        call.generate_states(
            clip_detections=clip_detections,
            config=expected_config,
        ),
        call.run_viterbi(
            candidate_states_by_frame=(
                expected_candidate_states
            ),
            clip_detections=clip_detections,
            config=expected_config,
        ),
        call.build_sequences(
            clip_detections=clip_detections,
            frame_player_state_sequence=(
                expected_state_sequence
            ),
            config=expected_config,
        ),
        call.interpolate(
            clip_player_pose_sequences=(
                expected_pose_sequences
            ),
            config=expected_config,
        ),
        call.assess_quality(
            clip_player_pose_sequences=(
                expected_pose_sequences
            ),
            config=expected_config,
        ),
    ]


def test_detect_players_assesses_interpolated_pose_sequence_object(
    mocker,
) -> None:
    clip_detections = _create_clip_detections()

    expected_config = PlayerDetectionConfig()
    expected_pose_sequences = mocker.Mock()
    expected_quality_report = (
        _create_quality_report()
    )

    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "PlayerDetectionConfig",
        return_value=expected_config,
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "generate_candidate_states_for_clip",
        return_value=[[(-1, -1)]],
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "viterbi_algorithm",
        return_value=[(-1, -1)],
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "build_two_player_pose_sequences",
        return_value=expected_pose_sequences,
    )

    pose_sequence_was_interpolated = False

    def interpolation_side_effect(
        *,
        clip_player_pose_sequences,
        config,
    ) -> None:
        nonlocal pose_sequence_was_interpolated

        assert (
            clip_player_pose_sequences
            is expected_pose_sequences
        )
        assert config is expected_config

        pose_sequence_was_interpolated = True

    def quality_side_effect(
        *,
        clip_player_pose_sequences,
        config,
    ) -> PoseSequenceQualityReport:
        assert pose_sequence_was_interpolated is True
        assert (
            clip_player_pose_sequences
            is expected_pose_sequences
        )
        assert config is expected_config

        return expected_quality_report

    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "interpolate_two_player_pose_sequences_in_place",
        side_effect=interpolation_side_effect,
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "assess_pose_sequence_quality",
        side_effect=quality_side_effect,
    )

    result = detect_players(
        clip_detections=clip_detections,
    )

    assert pose_sequence_was_interpolated is True
    assert result == (
        expected_pose_sequences,
        expected_quality_report,
    )


def test_detect_players_propagates_viterbi_failure_and_stops_pipeline(
    mocker,
) -> None:
    clip_detections = _create_clip_detections()

    expected_config = PlayerDetectionConfig()
    expected_candidate_states = [
        [
            (-1, -1),
        ],
    ]

    viterbi_error = RuntimeError(
        "Unable to determine player state sequence"
    )

    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "PlayerDetectionConfig",
        return_value=expected_config,
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "generate_candidate_states_for_clip",
        return_value=expected_candidate_states,
    )
    viterbi_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "viterbi_algorithm",
        side_effect=viterbi_error,
    )
    build_sequences_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "build_two_player_pose_sequences",
    )
    interpolation_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "interpolate_two_player_pose_sequences_in_place",
    )
    quality_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "assess_pose_sequence_quality",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Unable to determine player state sequence"
        ),
    ) as exception_info:
        detect_players(
            clip_detections=clip_detections,
        )

    assert exception_info.value is viterbi_error

    viterbi_mock.assert_called_once_with(
        candidate_states_by_frame=(
            expected_candidate_states
        ),
        clip_detections=clip_detections,
        config=expected_config,
    )
    build_sequences_mock.assert_not_called()
    interpolation_mock.assert_not_called()
    quality_mock.assert_not_called()


def test_detect_players_propagates_interpolation_failure_without_assessing_quality(
    mocker,
) -> None:
    clip_detections = _create_clip_detections()

    expected_config = PlayerDetectionConfig()
    expected_pose_sequences = mocker.Mock()

    interpolation_error = RuntimeError(
        "Pose interpolation failed"
    )

    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "PlayerDetectionConfig",
        return_value=expected_config,
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "generate_candidate_states_for_clip",
        return_value=[[(-1, -1)]],
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "viterbi_algorithm",
        return_value=[(-1, -1)],
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "build_two_player_pose_sequences",
        return_value=expected_pose_sequences,
    )
    interpolation_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "interpolate_two_player_pose_sequences_in_place",
        side_effect=interpolation_error,
    )
    quality_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "assess_pose_sequence_quality",
    )

    with pytest.raises(
        RuntimeError,
        match="Pose interpolation failed",
    ) as exception_info:
        detect_players(
            clip_detections=clip_detections,
        )

    assert exception_info.value is interpolation_error

    interpolation_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            expected_pose_sequences
        ),
        config=expected_config,
    )
    quality_mock.assert_not_called()


def test_detect_players_propagates_quality_assessment_failure(
    mocker,
) -> None:
    clip_detections = _create_clip_detections()

    expected_config = PlayerDetectionConfig()
    expected_pose_sequences = mocker.Mock()

    quality_error = ValueError(
        "Invalid pose sequence shape"
    )

    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "PlayerDetectionConfig",
        return_value=expected_config,
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "generate_candidate_states_for_clip",
        return_value=[[(-1, -1)]],
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "viterbi_algorithm",
        return_value=[(-1, -1)],
    )
    mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "build_two_player_pose_sequences",
        return_value=expected_pose_sequences,
    )
    interpolation_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "interpolate_two_player_pose_sequences_in_place",
    )
    quality_mock = mocker.patch(
        f"{PLAYER_DETECTION_PIPELINE_MODULE_PATH}."
        "assess_pose_sequence_quality",
        side_effect=quality_error,
    )

    with pytest.raises(
        ValueError,
        match="Invalid pose sequence shape",
    ) as exception_info:
        detect_players(
            clip_detections=clip_detections,
        )

    assert exception_info.value is quality_error

    interpolation_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            expected_pose_sequences
        ),
        config=expected_config,
    )
    quality_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            expected_pose_sequences
        ),
        config=expected_config,
    )
