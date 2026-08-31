from unittest.mock import call

import math

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    FrameDetections,
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
)
from ai_judo_coach.inference.player_detection.scoring.transition_score import (
    transition_score,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


TRANSITION_SCORE_MODULE_PATH = (
    "ai_judo_coach.inference.player_detection."
    "scoring.transition_score"
)


def _create_person_detection(
    detection_idx: int,
    track_id: int | None,
) -> PersonDetection:
    """Create one person detection for transition-scoring tests."""

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
    track_ids: list[int | None],
    frame_idx: int,
) -> FrameDetections:
    """Create one frame with detections using the supplied track IDs."""

    return FrameDetections(
        person_detections=[
            _create_person_detection(
                detection_idx=detection_idx,
                track_id=track_id,
            )
            for detection_idx, track_id
            in enumerate(track_ids)
        ],
        frame_idx=frame_idx,
        frame_shape_hw=(1080, 1920),
    )


def test_transition_score_sums_both_player_transition_scores(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[10, 20],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[20, 10],
        frame_idx=1,
    )

    previous_detection_0 = (
        previous_frame.person_detections[0]
    )
    previous_detection_1 = (
        previous_frame.person_detections[1]
    )
    current_detection_0 = (
        current_frame.person_detections[0]
    )
    current_detection_1 = (
        current_frame.person_detections[1]
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        side_effect=[
            math.sqrt(2.0) * 0.25,
            math.sqrt(2.0) * 0.50,
        ],
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.4,
        bbox_center_distance_penalty_weight=0.5,
    )

    result = transition_score(
        previous_state=(0, 1),
        current_state=(1, 0),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    # Player 0: 0.4 - (0.25 * 0.5) = 0.275
    # Player 1: 0.4 - (0.50 * 0.5) = 0.150
    assert isinstance(result, float)
    assert result == pytest.approx(0.425)

    assert distance_mock.call_args_list == [
        call(
            person_detection_a=previous_detection_0,
            person_detection_b=current_detection_1,
        ),
        call(
            person_detection_a=previous_detection_1,
            person_detection_b=current_detection_0,
        ),
    ]


def test_transition_score_uses_assignment_indices_to_select_detections(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[10, 20],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[30, 40],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=0.0,
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.0,
        bbox_center_distance_penalty_weight=0.5,
    )

    result = transition_score(
        previous_state=(1, 0),
        current_state=(0, 1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    assert result == pytest.approx(0.0)

    assert distance_mock.call_args_list == [
        call(
            person_detection_a=(
                previous_frame.person_detections[1]
            ),
            person_detection_b=(
                current_frame.person_detections[0]
            ),
        ),
        call(
            person_detection_a=(
                previous_frame.person_detections[0]
            ),
            person_detection_b=(
                current_frame.person_detections[1]
            ),
        ),
    ]


def test_transition_score_scores_remaining_player_when_other_is_missing(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[7],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[8, 7],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0) * 0.5,
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.4,
        bbox_center_distance_penalty_weight=0.5,
    )

    result = transition_score(
        previous_state=(0, -1),
        current_state=(1, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    # Only player 0 contributes:
    # 0.4 - (0.5 * 0.5) = 0.15
    assert result == pytest.approx(0.15)

    distance_mock.assert_called_once_with(
        person_detection_a=(
            previous_frame.person_detections[0]
        ),
        person_detection_b=(
            current_frame.person_detections[1]
        ),
    )


@pytest.mark.parametrize(
    (
        "previous_state",
        "current_state",
    ),
    [
        (
            (-1, -1),
            (-1, -1),
        ),
        (
            (-1, 0),
            (0, -1),
        ),
        (
            (0, -1),
            (-1, 0),
        ),
    ],
)
def test_transition_score_returns_zero_when_each_player_has_a_missing_endpoint(
    mocker,
    previous_state: CandidateState,
    current_state: CandidateState,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
    )

    result = transition_score(
        previous_state=previous_state,
        current_state=current_state,
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=PlayerDetectionConfig(),
    )

    assert result == pytest.approx(0.0)
    distance_mock.assert_not_called()


@pytest.mark.parametrize(
    (
        "previous_track_id",
        "current_track_id",
        "expected_score",
    ),
    [
        (
            12,
            12,
            0.4,
        ),
        (
            12,
            13,
            0.0,
        ),
        (
            None,
            12,
            0.0,
        ),
        (
            12,
            None,
            0.0,
        ),
        (
            None,
            None,
            0.0,
        ),
    ],
)
def test_transition_score_adds_bonus_only_for_same_known_track_id(
    mocker,
    previous_track_id: int | None,
    current_track_id: int | None,
    expected_score: float,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[previous_track_id],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[current_track_id],
        frame_idx=1,
    )

    mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=0.0,
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.4,
        bbox_center_distance_penalty_weight=0.5,
    )

    result = transition_score(
        previous_state=(0, -1),
        current_state=(0, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    assert result == pytest.approx(
        expected_score
    )


@pytest.mark.parametrize(
    (
        "normalized_distance",
        "expected_score",
    ),
    [
        (
            0.0,
            0.0,
        ),
        (
            math.sqrt(2.0) * 0.25,
            -0.15,
        ),
        (
            math.sqrt(2.0) * 0.50,
            -0.30,
        ),
        (
            math.sqrt(2.0),
            -0.60,
        ),
        (
            math.sqrt(2.0) * 2.0,
            -0.60,
        ),
        (
            -1.0,
            0.0,
        ),
    ],
)
def test_transition_score_normalizes_and_clamps_bbox_distance_penalty(
    mocker,
    normalized_distance: float,
    expected_score: float,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[None],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[None],
        frame_idx=1,
    )

    mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=normalized_distance,
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.4,
        bbox_center_distance_penalty_weight=0.6,
    )

    result = transition_score(
        previous_state=(0, -1),
        current_state=(0, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    assert result == pytest.approx(
        expected_score
    )


def test_transition_score_can_be_negative_when_movement_penalty_exceeds_bonus(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[5],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[5],
        frame_idx=1,
    )

    mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0),
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.2,
        bbox_center_distance_penalty_weight=0.7,
    )

    result = transition_score(
        previous_state=(0, -1),
        current_state=(0, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    assert result == pytest.approx(-0.5)


def test_transition_score_returns_only_track_bonus_when_distance_weight_is_zero(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[5],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[5],
        frame_idx=1,
    )

    mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=math.sqrt(2.0),
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.4,
        bbox_center_distance_penalty_weight=0.0,
    )

    result = transition_score(
        previous_state=(0, -1),
        current_state=(0, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    assert result == pytest.approx(0.4)


def test_transition_score_uses_configured_missing_sentinel(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[20],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[20],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=0.0,
    )

    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
        same_track_id_bonus=0.4,
    )

    result = transition_score(
        previous_state=(-99, 0),
        current_state=(-99, 0),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
    )

    assert result == pytest.approx(0.4)

    distance_mock.assert_called_once_with(
        person_detection_a=(
            previous_frame.person_detections[0]
        ),
        person_detection_b=(
            current_frame.person_detections[0]
        ),
    )


def test_transition_score_propagates_invalid_previous_detection_index(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
    )

    with pytest.raises(IndexError):
        transition_score(
            previous_state=(5, -1),
            current_state=(0, -1),
            previous_frame_detections=previous_frame,
            current_frame_detections=current_frame,
            config=PlayerDetectionConfig(),
        )

    distance_mock.assert_not_called()


def test_transition_score_propagates_invalid_current_detection_index(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
    )

    with pytest.raises(IndexError):
        transition_score(
            previous_state=(0, -1),
            current_state=(5, -1),
            previous_frame_detections=previous_frame,
            current_frame_detections=current_frame,
            config=PlayerDetectionConfig(),
        )

    distance_mock.assert_not_called()


def test_transition_score_reuses_cached_single_player_scores(
    mocker,
) -> None:
    previous_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=0,
    )
    current_frame = _create_frame_detections(
        track_ids=[10],
        frame_idx=1,
    )

    distance_mock = mocker.patch(
        f"{TRANSITION_SCORE_MODULE_PATH}."
        "normalized_distance_between_bbox_centers",
        return_value=0.0,
    )

    config = PlayerDetectionConfig(
        same_track_id_bonus=0.4,
        bbox_center_distance_penalty_weight=0.5,
    )
    score_cache: dict[
        tuple[int, int],
        float,
    ] = {}

    first_result = transition_score(
        previous_state=(0, -1),
        current_state=(0, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
        single_player_score_cache=score_cache,
    )
    second_result = transition_score(
        previous_state=(0, -1),
        current_state=(0, -1),
        previous_frame_detections=previous_frame,
        current_frame_detections=current_frame,
        config=config,
        single_player_score_cache=score_cache,
    )

    assert first_result == pytest.approx(0.4)
    assert second_result == pytest.approx(0.4)

    distance_mock.assert_called_once()

    assert score_cache.keys() == {
        (0, 0),
        (-1, -1),
    }
    assert score_cache[(0, 0)] == pytest.approx(
        0.4
    )
    assert score_cache[(-1, -1)] == pytest.approx(
        0.0
    )
