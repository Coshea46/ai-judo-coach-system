import pytest

from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
)
from ai_judo_coach.inference.player_detection.scoring.missing_score import (
    missing_state_penalty,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


@pytest.mark.parametrize(
    "state",
    [
        (0, 1),
        (1, 0),
        (2, 5),
    ],
)
def test_missing_state_penalty_returns_zero_when_neither_player_is_missing(
    state: CandidateState,
) -> None:
    result = missing_state_penalty(
        state=state,
        config=PlayerDetectionConfig(),
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "state",
    [
        (-1, 0),
        (-1, 5),
        (0, -1),
        (5, -1),
    ],
)
def test_missing_state_penalty_returns_one_player_penalty(
    state: CandidateState,
) -> None:
    config = PlayerDetectionConfig(
        one_player_missing_penalty=0.37,
        both_players_missing_penalty=0.81,
    )

    result = missing_state_penalty(
        state=state,
        config=config,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.37)


def test_missing_state_penalty_returns_both_players_penalty() -> None:
    config = PlayerDetectionConfig(
        one_player_missing_penalty=0.37,
        both_players_missing_penalty=0.81,
    )

    result = missing_state_penalty(
        state=(-1, -1),
        config=config,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.81)


@pytest.mark.parametrize(
    (
        "state",
        "expected_penalty",
    ),
    [
        (
            (0, 1),
            0.0,
        ),
        (
            (-99, 1),
            0.25,
        ),
        (
            (1, -99),
            0.25,
        ),
        (
            (-99, -99),
            0.75,
        ),
    ],
)
def test_missing_state_penalty_uses_configured_missing_sentinel(
    state: CandidateState,
    expected_penalty: float,
) -> None:
    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
        one_player_missing_penalty=0.25,
        both_players_missing_penalty=0.75,
    )

    result = missing_state_penalty(
        state=state,
        config=config,
    )

    assert result == pytest.approx(
        expected_penalty
    )


@pytest.mark.parametrize(
    "state",
    [
        (-1, 0),
        (0, -1),
        (-1, -1),
    ],
)
def test_missing_state_penalty_does_not_treat_other_negative_indices_as_missing(
    state: CandidateState,
) -> None:
    config = PlayerDetectionConfig(
        missing_detection_sentinel=-99,
        one_player_missing_penalty=0.25,
        both_players_missing_penalty=0.75,
    )

    result = missing_state_penalty(
        state=state,
        config=config,
    )

    assert result == pytest.approx(0.0)


def test_missing_state_penalty_prefers_both_players_penalty_when_both_missing() -> None:
    config = PlayerDetectionConfig(
        one_player_missing_penalty=0.2,
        both_players_missing_penalty=0.9,
    )

    result = missing_state_penalty(
        state=(
            config.missing_detection_sentinel,
            config.missing_detection_sentinel,
        ),
        config=config,
    )

    assert result == pytest.approx(0.9)
    assert result != pytest.approx(
        config.one_player_missing_penalty
    )


@pytest.mark.parametrize(
    (
        "one_player_penalty",
        "both_players_penalty",
    ),
    [
        (0.0, 0.0),
        (0.1, 0.4),
        (1.0, 2.0),
    ],
)
def test_missing_state_penalty_returns_configured_penalty_values(
    one_player_penalty: float,
    both_players_penalty: float,
) -> None:
    config = PlayerDetectionConfig(
        one_player_missing_penalty=(
            one_player_penalty
        ),
        both_players_missing_penalty=(
            both_players_penalty
        ),
    )

    sentinel = config.missing_detection_sentinel

    one_missing_result = missing_state_penalty(
        state=(sentinel, 0),
        config=config,
    )
    both_missing_result = missing_state_penalty(
        state=(sentinel, sentinel),
        config=config,
    )

    assert one_missing_result == pytest.approx(
        one_player_penalty
    )
    assert both_missing_result == pytest.approx(
        both_players_penalty
    )
