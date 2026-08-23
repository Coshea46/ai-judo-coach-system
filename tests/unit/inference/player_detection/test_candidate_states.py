import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
    generate_candidate_states_for_clip,
    generate_candidate_states_for_frame,
    is_assignment_missing,
    state_has_both_players_missing,
    state_has_duplicate_detection,
    state_has_missing_player,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


def _create_person_detection(
    detection_idx: int,
) -> PersonDetection:
    """Create one person detection for candidate-state tests."""

    return PersonDetection(
        detection_idx=detection_idx,
        track_id=detection_idx + 10,
        bbox_xyxy_px=np.array(
            [10.0, 20.0, 50.0, 80.0],
            dtype=np.float32,
        ),
        bbox_xyxy_normalized=np.array(
            [0.1, 0.2, 0.5, 0.8],
            dtype=np.float32,
        ),
        bbox_conf=0.9,
        keypoints_xy_px=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_xy_norm=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_conf=np.zeros(
            17,
            dtype=np.float32,
        ),
    )


def _create_frame_detections(
    detection_count: int,
    frame_idx: int = 0,
) -> FrameDetections:
    """Create one frame containing the requested detections."""

    return FrameDetections(
        person_detections=[
            _create_person_detection(
                detection_idx=detection_idx,
            )
            for detection_idx in range(
                detection_count
            )
        ],
        frame_idx=frame_idx,
        frame_shape_hw=(1080, 1920),
    )


def test_generate_candidate_states_for_empty_frame_returns_both_missing() -> None:
    frame_detections = _create_frame_detections(
        detection_count=0,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        (-1, -1),
    ]


def test_generate_candidate_states_for_one_detection() -> None:
    frame_detections = _create_frame_detections(
        detection_count=1,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        (0, -1),
        (-1, 0),
        (-1, -1),
    ]


def test_generate_candidate_states_for_two_detections() -> None:
    frame_detections = _create_frame_detections(
        detection_count=2,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        (0, 1),
        (1, 0),
        (0, -1),
        (1, -1),
        (-1, 0),
        (-1, 1),
        (-1, -1),
    ]


def test_generate_candidate_states_for_three_detections() -> None:
    frame_detections = _create_frame_detections(
        detection_count=3,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (0, -1),
        (1, -1),
        (2, -1),
        (-1, 0),
        (-1, 1),
        (-1, 2),
        (-1, -1),
    ]


@pytest.mark.parametrize(
    (
        "detection_count",
        "expected_state_count",
    ),
    [
        (0, 1),
        (1, 3),
        (2, 7),
        (3, 13),
        (4, 21),
        (5, 31),
    ],
)
def test_generate_candidate_states_returns_expected_number_of_states(
    detection_count: int,
    expected_state_count: int,
) -> None:
    frame_detections = _create_frame_detections(
        detection_count=detection_count,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert len(result) == expected_state_count


@pytest.mark.parametrize(
    "detection_count",
    [
        1,
        2,
        3,
        4,
    ],
)
def test_generate_candidate_states_returns_unique_states(
    detection_count: int,
) -> None:
    frame_detections = _create_frame_detections(
        detection_count=detection_count,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert len(result) == len(set(result))


def test_generate_candidate_states_never_assigns_same_detection_to_both_players() -> None:
    frame_detections = _create_frame_detections(
        detection_count=4,
    )
    config = PlayerDetectionConfig()

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=config,
    )

    assert all(
        not state_has_duplicate_detection(
            state=state,
            config=config,
        )
        for state in result
    )


def test_generate_candidate_states_includes_every_ordered_detection_pair() -> None:
    frame_detections = _create_frame_detections(
        detection_count=3,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    both_assigned_states = {
        state
        for state in result
        if -1 not in state
    }

    assert both_assigned_states == {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }


def test_generate_candidate_states_includes_each_single_missing_assignment() -> None:
    frame_detections = _create_frame_detections(
        detection_count=3,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert {
        (0, -1),
        (1, -1),
        (2, -1),
    }.issubset(result)

    assert {
        (-1, 0),
        (-1, 1),
        (-1, 2),
    }.issubset(result)


def test_generate_candidate_states_uses_configured_missing_sentinel() -> None:
    frame_detections = _create_frame_detections(
        detection_count=2,
    )

    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=config,
    )

    assert result == [
        (0, 1),
        (1, 0),
        (0, -99),
        (1, -99),
        (-99, 0),
        (-99, 1),
        (-99, -99),
    ]


def test_generate_candidate_states_for_empty_frame_uses_configured_sentinel() -> None:
    frame_detections = _create_frame_detections(
        detection_count=0,
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(
            missing_detection_sentinel=-99,
        ),
    )

    assert result == [
        (-99, -99),
    ]


def test_generate_candidate_states_uses_list_positions_as_assignment_indices() -> None:
    frame_detections = FrameDetections(
        person_detections=[
            _create_person_detection(
                detection_idx=12,
            ),
            _create_person_detection(
                detection_idx=35,
            ),
        ],
        frame_idx=0,
        frame_shape_hw=(1080, 1920),
    )

    result = generate_candidate_states_for_frame(
        frame_detections=frame_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        (0, 1),
        (1, 0),
        (0, -1),
        (1, -1),
        (-1, 0),
        (-1, 1),
        (-1, -1),
    ]


def test_generate_candidate_states_for_clip_processes_every_frame() -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                detection_count=0,
                frame_idx=0,
            ),
            _create_frame_detections(
                detection_count=1,
                frame_idx=1,
            ),
            _create_frame_detections(
                detection_count=2,
                frame_idx=2,
            ),
        ],
        clip_id="clip_0",
    )

    result = generate_candidate_states_for_clip(
        clip_detections=clip_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == [
        [
            (-1, -1),
        ],
        [
            (0, -1),
            (-1, 0),
            (-1, -1),
        ],
        [
            (0, 1),
            (1, 0),
            (0, -1),
            (1, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1),
        ],
    ]


def test_generate_candidate_states_for_clip_returns_empty_list_for_empty_clip() -> None:
    clip_detections = ClipDetections(
        frame_detections=[],
        clip_id="empty_clip",
    )

    result = generate_candidate_states_for_clip(
        clip_detections=clip_detections,
        config=PlayerDetectionConfig(),
    )

    assert result == []


def test_generate_candidate_states_for_clip_uses_configured_sentinel() -> None:
    clip_detections = ClipDetections(
        frame_detections=[
            _create_frame_detections(
                detection_count=0,
                frame_idx=0,
            ),
            _create_frame_detections(
                detection_count=1,
                frame_idx=1,
            ),
        ],
        clip_id="clip_0",
    )

    result = generate_candidate_states_for_clip(
        clip_detections=clip_detections,
        config=PlayerDetectionConfig(
            missing_detection_sentinel=-99,
        ),
    )

    assert result == [
        [
            (-99, -99),
        ],
        [
            (0, -99),
            (-99, 0),
            (-99, -99),
        ],
    ]


@pytest.mark.parametrize(
    (
        "assignment_idx",
        "expected_result",
    ),
    [
        (-1, True),
        (0, False),
        (1, False),
        (-2, False),
        (100, False),
    ],
)
def test_is_assignment_missing_uses_default_sentinel(
    assignment_idx: int,
    expected_result: bool,
) -> None:
    result = is_assignment_missing(
        assignment_idx=assignment_idx,
        config=PlayerDetectionConfig(),
    )

    assert result is expected_result


@pytest.mark.parametrize(
    (
        "assignment_idx",
        "expected_result",
    ),
    [
        (-99, True),
        (-1, False),
        (0, False),
        (99, False),
    ],
)
def test_is_assignment_missing_uses_configured_sentinel(
    assignment_idx: int,
    expected_result: bool,
) -> None:
    result = is_assignment_missing(
        assignment_idx=assignment_idx,
        config=PlayerDetectionConfig(
            missing_detection_sentinel=-99,
        ),
    )

    assert result is expected_result


@pytest.mark.parametrize(
    (
        "state",
        "expected_result",
    ),
    [
        ((0, 1), False),
        ((1, 0), False),
        ((-1, 0), True),
        ((0, -1), True),
        ((-1, -1), True),
        ((-2, 0), False),
    ],
)
def test_state_has_missing_player(
    state: CandidateState,
    expected_result: bool,
) -> None:
    result = state_has_missing_player(
        state=state,
        config=PlayerDetectionConfig(),
    )

    assert result is expected_result


@pytest.mark.parametrize(
    (
        "state",
        "expected_result",
    ),
    [
        ((0, 1), False),
        ((-1, 0), False),
        ((0, -1), False),
        ((-1, -1), True),
        ((-2, -2), False),
        ((-1, -2), False),
    ],
)
def test_state_has_both_players_missing(
    state: CandidateState,
    expected_result: bool,
) -> None:
    result = state_has_both_players_missing(
        state=state,
        config=PlayerDetectionConfig(),
    )

    assert result is expected_result


@pytest.mark.parametrize(
    (
        "state",
        "expected_result",
    ),
    [
        ((0, 0), True),
        ((1, 1), True),
        ((0, 1), False),
        ((1, 0), False),
        ((-1, 0), False),
        ((0, -1), False),
        ((-1, -1), False),
    ],
)
def test_state_has_duplicate_detection(
    state: CandidateState,
    expected_result: bool,
) -> None:
    result = state_has_duplicate_detection(
        state=state,
        config=PlayerDetectionConfig(),
    )

    assert result is expected_result


def test_state_helpers_use_configured_missing_sentinel() -> None:
    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
    )

    assert state_has_missing_player(
        state=(-99, 0),
        config=config,
    )

    assert state_has_missing_player(
        state=(0, -99),
        config=config,
    )

    assert state_has_both_players_missing(
        state=(-99, -99),
        config=config,
    )

    assert not state_has_both_players_missing(
        state=(-99, 0),
        config=config,
    )

    assert not state_has_duplicate_detection(
        state=(-99, -99),
        config=config,
    )

    assert state_has_duplicate_detection(
        state=(2, 2),
        config=config,
    )
