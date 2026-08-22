"""
This file contains functions for computing
heuristic scores for candidate player-assignment
states in a single frame.

A state score combines:
- individual detection scores for assigned players
- pair score when both players are assigned
- missing-player penalties

Higher values indicate more plausible states.
"""

from ai_judo_coach.inference.inference_schemas import FrameDetections
from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
    is_assignment_missing,
    state_has_missing_player,
)
from .detection_score import detection_score
from .missing_score import missing_state_penalty
from .pair_score import pair_score
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


def state_score(
    state: CandidateState,
    frame_detections: FrameDetections,
    config: PlayerDetectionConfig,
) -> float:
    """
    Computes a heuristic score for a candidate
    player-assignment state in a single frame.

    Higher values indicate a more plausible state.

    State is a tuple of the form:
    (player_0_detection_idx, player_1_detection_idx)
    """

    missing_player = state_has_missing_player(
        state=state,
        config=config
    )

    player_0_assignment_idx = state[0]
    player_1_assignment_idx = state[1]

    person_a = None
    person_b = None

    person_a_detection_score = 0.0
    person_b_detection_score = 0.0
    player_pair_score = 0.0

    missing_players_penalty = missing_state_penalty(
        state=state,
        config=config
    )

    if not is_assignment_missing(
        assignment_idx=player_0_assignment_idx,
        config=config
    ):
        person_a = frame_detections.person_detections[
            player_0_assignment_idx
        ]

        person_a_detection_score = detection_score(
            person_detection=person_a,
            config=config
        )

    if not is_assignment_missing(
        assignment_idx=player_1_assignment_idx,
        config=config
    ):
        person_b = frame_detections.person_detections[
            player_1_assignment_idx
        ]

        person_b_detection_score = detection_score(
            person_detection=person_b,
            config=config
        )

    if not missing_player:
        player_pair_score = pair_score(
            person_detection_a=person_a,
            person_detection_b=person_b,
            config=config
        )

    # sum scores together before applying any penalties
    initial_score_for_state = (
        person_a_detection_score
        + person_b_detection_score
        + player_pair_score
    )

    # now subtract any penalties from the initial score
    final_score_for_state = (
        initial_score_for_state
        - missing_players_penalty
    )

    return float(final_score_for_state)



