from unittest.mock import call

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    FrameDetections,
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.scoring.state_score import (
    state_score,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


STATE_SCORE_MODULE_PATH = (
    "ai_judo_coach.inference.player_detection."
    "scoring.state_score"
)


def _create_person_detection(
    detection_idx: int,
) -> PersonDetection:
    """Create one person detection for state-scoring tests."""

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
    detection_count: int = 2,
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
        frame_idx=0,
        frame_shape_hw=(1080, 1920),
    )


def test_state_score_combines_two_detection_scores_and_pair_score(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=2,
        )
    )

    detection_a = (
        frame_detections.person_detections[0]
    )
    detection_b = (
        frame_detections.person_detections[1]
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
        side_effect=[
            0.6,
            0.7,
        ],
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
        return_value=0.8,
    )

    config = PlayerDetectionConfig()

    result = state_score(
        state=(0, 1),
        frame_detections=frame_detections,
        config=config,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        0.6 + 0.7 + 0.8
    )

    assert detection_score_mock.call_args_list == [
        call(
            person_detection=detection_a,
            config=config,
        ),
        call(
            person_detection=detection_b,
            config=config,
        ),
    ]

    pair_score_mock.assert_called_once_with(
        person_detection_a=detection_a,
        person_detection_b=detection_b,
        config=config,
    )


def test_state_score_uses_assignment_indices_to_select_detections(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=2,
        )
    )

    detection_a = (
        frame_detections.person_detections[0]
    )
    detection_b = (
        frame_detections.person_detections[1]
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
        side_effect=[
            0.4,
            0.9,
        ],
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
        return_value=0.5,
    )

    config = PlayerDetectionConfig()

    result = state_score(
        state=(1, 0),
        frame_detections=frame_detections,
        config=config,
    )

    assert result == pytest.approx(
        0.4 + 0.9 + 0.5
    )

    assert detection_score_mock.call_args_list == [
        call(
            person_detection=detection_b,
            config=config,
        ),
        call(
            person_detection=detection_a,
            config=config,
        ),
    ]

    pair_score_mock.assert_called_once_with(
        person_detection_a=detection_b,
        person_detection_b=detection_a,
        config=config,
    )


@pytest.mark.parametrize(
    (
        "state",
        "assigned_detection_index",
    ),
    [
        (
            (-1, 1),
            1,
        ),
        (
            (0, -1),
            0,
        ),
    ],
)
def test_state_score_scores_only_assigned_detection_when_one_player_is_missing(
    mocker,
    state: tuple[int, int],
    assigned_detection_index: int,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=2,
        )
    )

    assigned_detection = (
        frame_detections.person_detections[
            assigned_detection_index
        ]
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
        return_value=0.75,
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
    )

    config = PlayerDetectionConfig(
        one_player_missing_penalty=0.3,
    )

    result = state_score(
        state=state,
        frame_detections=frame_detections,
        config=config,
    )

    assert result == pytest.approx(
        0.75 - 0.3
    )

    detection_score_mock.assert_called_once_with(
        person_detection=assigned_detection,
        config=config,
    )
    pair_score_mock.assert_not_called()


def test_state_score_returns_negative_penalty_when_both_players_are_missing(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=0,
        )
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
    )

    config = PlayerDetectionConfig(
        both_players_missing_penalty=0.65,
    )

    sentinel = (
        config.missing_detection_sentinel
    )

    result = state_score(
        state=(sentinel, sentinel),
        frame_detections=frame_detections,
        config=config,
    )

    assert result == pytest.approx(-0.65)
    detection_score_mock.assert_not_called()
    pair_score_mock.assert_not_called()


@pytest.mark.parametrize(
    (
        "state",
        "expected_score",
    ),
    [
        (
            (-99, 1),
            0.7 - 0.4,
        ),
        (
            (0, -99),
            0.7 - 0.4,
        ),
    ],
)
def test_state_score_uses_configured_missing_sentinel_for_one_player(
    mocker,
    state: tuple[int, int],
    expected_score: float,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=2,
        )
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
        return_value=0.7,
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
    )

    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
        one_player_missing_penalty=0.4,
    )

    result = state_score(
        state=state,
        frame_detections=frame_detections,
        config=config,
    )

    assert result == pytest.approx(
        expected_score
    )
    detection_score_mock.assert_called_once()
    pair_score_mock.assert_not_called()


def test_state_score_uses_configured_missing_sentinel_for_both_players(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=0,
        )
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
    )

    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
        both_players_missing_penalty=0.85,
    )

    result = state_score(
        state=(-99, -99),
        frame_detections=frame_detections,
        config=config,
    )

    assert result == pytest.approx(-0.85)
    detection_score_mock.assert_not_called()
    pair_score_mock.assert_not_called()


def test_state_score_can_be_negative_when_missing_penalty_exceeds_detection_score(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=1,
        )
    )

    mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
        return_value=0.1,
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
    )

    config = PlayerDetectionConfig(
        one_player_missing_penalty=0.3,
    )

    result = state_score(
        state=(0, -1),
        frame_detections=frame_detections,
        config=config,
    )

    assert result == pytest.approx(-0.2)
    pair_score_mock.assert_not_called()


def test_state_score_propagates_invalid_detection_index(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=1,
        )
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
    )

    with pytest.raises(IndexError):
        state_score(
            state=(5, -1),
            frame_detections=frame_detections,
            config=PlayerDetectionConfig(),
        )

    detection_score_mock.assert_not_called()
    pair_score_mock.assert_not_called()


def test_state_score_reuses_detection_and_symmetric_pair_scores(
    mocker,
) -> None:
    frame_detections = (
        _create_frame_detections(
            detection_count=2,
        )
    )

    detection_a = (
        frame_detections.person_detections[0]
    )
    detection_b = (
        frame_detections.person_detections[1]
    )

    detection_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "detection_score",
        side_effect=[
            0.6,
            0.7,
        ],
    )
    pair_score_mock = mocker.patch(
        f"{STATE_SCORE_MODULE_PATH}."
        "pair_score",
        return_value=0.8,
    )

    config = PlayerDetectionConfig()

    detection_score_cache: dict[
        int,
        float,
    ] = {}
    pair_score_cache: dict[
        tuple[int, int],
        float,
    ] = {}

    first_result = state_score(
        state=(0, 1),
        frame_detections=frame_detections,
        config=config,
        detection_score_cache=(
            detection_score_cache
        ),
        pair_score_cache=pair_score_cache,
    )
    second_result = state_score(
        state=(1, 0),
        frame_detections=frame_detections,
        config=config,
        detection_score_cache=(
            detection_score_cache
        ),
        pair_score_cache=pair_score_cache,
    )

    assert first_result == pytest.approx(
        0.6 + 0.7 + 0.8
    )
    assert second_result == pytest.approx(
        0.6 + 0.7 + 0.8
    )

    assert detection_score_mock.call_args_list == [
        call(
            person_detection=detection_a,
            config=config,
        ),
        call(
            person_detection=detection_b,
            config=config,
        ),
    ]

    pair_score_mock.assert_called_once_with(
        person_detection_a=detection_a,
        person_detection_b=detection_b,
        config=config,
    )

    assert detection_score_cache.keys() == {
        0,
        1,
    }
    assert detection_score_cache[0] == pytest.approx(
        0.6
    )
    assert detection_score_cache[1] == pytest.approx(
        0.7
    )

    assert pair_score_cache.keys() == {
        (0, 1),
    }
    assert pair_score_cache[(0, 1)] == pytest.approx(
        0.8
    )
