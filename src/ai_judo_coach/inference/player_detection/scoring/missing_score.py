"""
This file contains functions for computing
penalties associated with candidate states
where one or both players are missing.
"""

from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
    state_has_both_players_missing,
    state_has_missing_player,
)
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


def missing_state_penalty(
    state: CandidateState,
    config: PlayerDetectionConfig,
) -> float:
    """
    Computes the positive penalty magnitude associated
    with missing players in a candidate state.

    The returned value should be subtracted from the
    state score.

    Returns:
    - 0.0 if neither player is missing
    - config.one_player_missing_penalty if one player is missing
    - config.both_players_missing_penalty if both players are missing
    """

    no_penalty_value = 0.0

    if state_has_both_players_missing(state=state, config=config):
        return float(config.both_players_missing_penalty)

    if state_has_missing_player(state=state, config=config):
        return float(config.one_player_missing_penalty)


    return float(no_penalty_value)


    
