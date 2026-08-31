from unittest.mock import ANY, call

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
)
from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
)
from ai_judo_coach.inference.player_detection.solvers.viterbi import (
    viterbi_algorithm,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


VITERBI_MODULE_PATH = (
    "ai_judo_coach.inference.player_detection."
    "solvers.viterbi"
)


def _create_clip_detections(
    frame_count: int,
) -> ClipDetections:
    """Create a clip with the requested number of frames."""

    return ClipDetections(
        frame_detections=[
            FrameDetections(
                person_detections=[],
                frame_idx=frame_idx,
                frame_shape_hw=(1080, 1920),
            )
            for frame_idx in range(frame_count)
        ],
        clip_id="clip_0",
    )


def test_viterbi_algorithm_returns_empty_list_for_empty_clip() -> None:
    result = viterbi_algorithm(
        candidate_states_by_frame=[],
        clip_detections=_create_clip_detections(
            frame_count=0,
        ),
        config=PlayerDetectionConfig(),
    )

    assert result == []


def test_viterbi_algorithm_rejects_frame_count_mismatch(
    mocker,
) -> None:
    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score"
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Number of candidate-state frames must match "
            "number of frames in ClipDetections"
        ),
    ):
        viterbi_algorithm(
            candidate_states_by_frame=[
                [(0, -1)],
            ],
            clip_detections=_create_clip_detections(
                frame_count=2,
            ),
            config=PlayerDetectionConfig(),
        )

    state_score_mock.assert_not_called()
    transition_matrix_mock.assert_not_called()


def test_viterbi_algorithm_rejects_first_frame_without_candidate_states(
    mocker,
) -> None:
    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score"
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix"
    )

    with pytest.raises(
        ValueError,
        match="Frame 0 has no candidate states",
    ):
        viterbi_algorithm(
            candidate_states_by_frame=[
                [],
            ],
            clip_detections=_create_clip_detections(
                frame_count=1,
            ),
            config=PlayerDetectionConfig(),
        )

    state_score_mock.assert_not_called()
    transition_matrix_mock.assert_not_called()


def test_viterbi_algorithm_rejects_later_frame_without_candidate_states(
    mocker,
) -> None:
    first_state: CandidateState = (0, -1)

    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        return_value=0.5,
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix"
    )

    config = PlayerDetectionConfig()
    clip_detections = _create_clip_detections(
        frame_count=2,
    )

    with pytest.raises(
        ValueError,
        match="Frame 1 has no candidate states",
    ):
        viterbi_algorithm(
            candidate_states_by_frame=[
                [first_state],
                [],
            ],
            clip_detections=clip_detections,
            config=config,
        )

    state_score_mock.assert_called_once_with(
        state=first_state,
        frame_detections=(
            clip_detections.frame_detections[0]
        ),
        config=config,
        detection_score_cache=ANY,
        pair_score_cache=ANY,
    )
    transition_matrix_mock.assert_not_called()


def test_viterbi_algorithm_selects_highest_scoring_state_for_single_frame(
    mocker,
) -> None:
    first_state: CandidateState = (0, 1)
    second_state: CandidateState = (1, 0)
    third_state: CandidateState = (-1, -1)

    clip_detections = _create_clip_detections(
        frame_count=1,
    )
    frame_detections = (
        clip_detections.frame_detections[0]
    )

    state_scores = {
        first_state: 0.7,
        second_state: 1.4,
        third_state: -0.5,
    }

    def state_score_side_effect(
        *,
        state: CandidateState,
        frame_detections: FrameDetections,
        config: PlayerDetectionConfig,
        detection_score_cache: dict[
            int,
            float,
        ],
        pair_score_cache: dict[
            tuple[int, int],
            float,
        ],
    ) -> float:
        del frame_detections
        del config
        del detection_score_cache
        del pair_score_cache

        return state_scores[state]

    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        side_effect=state_score_side_effect,
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix"
    )

    config = PlayerDetectionConfig()

    result = viterbi_algorithm(
        candidate_states_by_frame=[
            [
                first_state,
                second_state,
                third_state,
            ],
        ],
        clip_detections=clip_detections,
        config=config,
    )

    assert result == [
        second_state,
    ]

    assert state_score_mock.call_args_list == [
        call(
            state=first_state,
            frame_detections=frame_detections,
            config=config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
        call(
            state=second_state,
            frame_detections=frame_detections,
            config=config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
        call(
            state=third_state,
            frame_detections=frame_detections,
            config=config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
    ]

    transition_matrix_mock.assert_not_called()


def test_viterbi_algorithm_selects_best_complete_path(
    mocker,
) -> None:
    frame_0_state_a: CandidateState = (0, -1)
    frame_0_state_b: CandidateState = (-1, 0)

    frame_1_state_c: CandidateState = (1, -1)
    frame_1_state_d: CandidateState = (-1, 1)

    candidate_states_by_frame = [
        [
            frame_0_state_a,
            frame_0_state_b,
        ],
        [
            frame_1_state_c,
            frame_1_state_d,
        ],
    ]

    clip_detections = _create_clip_detections(
        frame_count=2,
    )
    previous_frame = (
        clip_detections.frame_detections[0]
    )
    current_frame = (
        clip_detections.frame_detections[1]
    )

    state_scores = {
        (0, frame_0_state_a): 5.0,
        (0, frame_0_state_b): 0.0,
        (1, frame_1_state_c): 0.0,
        (1, frame_1_state_d): 2.0,
    }

    expected_config = PlayerDetectionConfig()

    def state_score_side_effect(
        *,
        state: CandidateState,
        frame_detections: FrameDetections,
        config: PlayerDetectionConfig,
        detection_score_cache: dict[
            int,
            float,
        ],
        pair_score_cache: dict[
            tuple[int, int],
            float,
        ],
    ) -> float:
        assert config is expected_config
        assert isinstance(
            detection_score_cache,
            dict,
        )
        assert isinstance(
            pair_score_cache,
            dict,
        )

        return state_scores[
            (
                frame_detections.frame_idx,
                state,
            )
        ]

    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        side_effect=state_score_side_effect,
    )

    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix",
        return_value=np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    10.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        ),
    )

    result = viterbi_algorithm(
        candidate_states_by_frame=(
            candidate_states_by_frame
        ),
        clip_detections=clip_detections,
        config=expected_config,
    )

    # The locally strongest state in frame 0 is state A, but the
    # transition from state B to state C makes B -> C the strongest
    # complete path.
    assert result == [
        frame_0_state_b,
        frame_1_state_c,
    ]

    assert state_score_mock.call_args_list == [
        call(
            state=frame_0_state_a,
            frame_detections=previous_frame,
            config=expected_config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
        call(
            state=frame_0_state_b,
            frame_detections=previous_frame,
            config=expected_config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
        call(
            state=frame_1_state_c,
            frame_detections=current_frame,
            config=expected_config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
        call(
            state=frame_1_state_d,
            frame_detections=current_frame,
            config=expected_config,
            detection_score_cache=ANY,
            pair_score_cache=ANY,
        ),
    ]

    transition_matrix_mock.assert_called_once_with(
        previous_states=[
            frame_0_state_a,
            frame_0_state_b,
        ],
        current_states=[
            frame_1_state_c,
            frame_1_state_d,
        ],
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=expected_config,
    )


def test_viterbi_algorithm_backtracks_across_multiple_frames(
    mocker,
) -> None:
    state_a: CandidateState = (0, -1)
    state_b: CandidateState = (-1, 0)

    candidate_states_by_frame = [
        [state_a, state_b],
        [state_a, state_b],
        [state_a, state_b],
    ]

    clip_detections = _create_clip_detections(
        frame_count=3,
    )

    state_scores = {
        (0, state_a): 1.0,
        (0, state_b): 0.0,
        (1, state_a): 0.0,
        (1, state_b): 0.0,
        (2, state_a): 0.0,
        (2, state_b): 0.0,
    }

    transition_matrices = [
        np.asarray(
            [
                [
                    0.0,
                    2.0,
                ],
                [
                    0.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    3.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        ),
    ]

    def state_score_side_effect(
        *,
        state: CandidateState,
        frame_detections: FrameDetections,
        config: PlayerDetectionConfig,
        detection_score_cache: dict[
            int,
            float,
        ],
        pair_score_cache: dict[
            tuple[int, int],
            float,
        ],
    ) -> float:
        del config
        del detection_score_cache
        del pair_score_cache

        return state_scores[
            (
                frame_detections.frame_idx,
                state,
            )
        ]

    mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        side_effect=state_score_side_effect,
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix",
        side_effect=transition_matrices,
    )

    result = viterbi_algorithm(
        candidate_states_by_frame=(
            candidate_states_by_frame
        ),
        clip_detections=clip_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        state_a,
        state_b,
        state_a,
    ]

    assert transition_matrix_mock.call_count == 2


def test_viterbi_algorithm_uses_first_state_when_final_scores_are_tied(
    mocker,
) -> None:
    first_state: CandidateState = (0, -1)
    second_state: CandidateState = (-1, 0)

    mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        return_value=1.0,
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix"
    )

    result = viterbi_algorithm(
        candidate_states_by_frame=[
            [
                first_state,
                second_state,
            ],
        ],
        clip_detections=_create_clip_detections(
            frame_count=1,
        ),
        config=PlayerDetectionConfig(),
    )

    assert result == [
        first_state,
    ]
    transition_matrix_mock.assert_not_called()


def test_viterbi_algorithm_uses_first_predecessor_when_path_scores_are_tied(
    mocker,
) -> None:
    first_predecessor: CandidateState = (0, -1)
    second_predecessor: CandidateState = (-1, 0)
    final_state: CandidateState = (0, 1)

    clip_detections = _create_clip_detections(
        frame_count=2,
    )

    mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        return_value=0.0,
    )
    mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix",
        return_value=np.zeros(
            (
                2,
                1,
            ),
            dtype=np.float64,
        ),
    )

    result = viterbi_algorithm(
        candidate_states_by_frame=[
            [
                first_predecessor,
                second_predecessor,
            ],
            [
                final_state,
            ],
        ],
        clip_detections=clip_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        first_predecessor,
        final_state,
    ]


def test_viterbi_algorithm_raises_when_no_finite_predecessor_can_be_selected(
    mocker,
) -> None:
    first_state: CandidateState = (0, -1)
    second_state: CandidateState = (1, -1)

    def state_score_side_effect(
        *,
        state: CandidateState,
        frame_detections: FrameDetections,
        config: PlayerDetectionConfig,
        detection_score_cache: dict[
            int,
            float,
        ],
        pair_score_cache: dict[
            tuple[int, int],
            float,
        ],
    ) -> float:
        del state
        del config
        del detection_score_cache
        del pair_score_cache

        if frame_detections.frame_idx == 0:
            return -float("inf")

        return 0.0

    mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        side_effect=state_score_side_effect,
    )
    mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix",
        return_value=np.zeros(
            (
                1,
                1,
            ),
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Could not find a predecessor for "
            r"state \(1, -1\) in frame 1"
        ),
    ):
        viterbi_algorithm(
            candidate_states_by_frame=[
                [
                    first_state,
                ],
                [
                    second_state,
                ],
            ],
            clip_detections=_create_clip_detections(
                frame_count=2,
            ),
            config=PlayerDetectionConfig(),
        )


def test_viterbi_algorithm_propagates_state_score_failure(
    mocker,
) -> None:
    scoring_error = RuntimeError(
        "State scoring failed"
    )

    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        side_effect=scoring_error,
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix"
    )

    config = PlayerDetectionConfig()
    clip_detections = _create_clip_detections(
        frame_count=1,
    )

    with pytest.raises(
        RuntimeError,
        match="State scoring failed",
    ) as exception_info:
        viterbi_algorithm(
            candidate_states_by_frame=[
                [
                    (0, -1),
                ],
            ],
            clip_detections=clip_detections,
            config=config,
        )

    assert exception_info.value is scoring_error
    state_score_mock.assert_called_once()
    transition_matrix_mock.assert_not_called()


def test_viterbi_algorithm_propagates_transition_matrix_failure(
    mocker,
) -> None:
    transition_error = RuntimeError(
        "Transition scoring failed"
    )

    state_score_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}.state_score",
        return_value=0.5,
    )
    transition_matrix_mock = mocker.patch(
        f"{VITERBI_MODULE_PATH}."
        "build_transition_score_matrix",
        side_effect=transition_error,
    )

    with pytest.raises(
        RuntimeError,
        match="Transition scoring failed",
    ) as exception_info:
        viterbi_algorithm(
            candidate_states_by_frame=[
                [
                    (0, -1),
                ],
                [
                    (1, -1),
                ],
            ],
            clip_detections=_create_clip_detections(
                frame_count=2,
            ),
            config=PlayerDetectionConfig(),
        )

    assert exception_info.value is transition_error
    assert state_score_mock.call_count == 2
    transition_matrix_mock.assert_called_once()
