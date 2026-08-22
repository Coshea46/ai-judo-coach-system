"""
This file contains functions that determine the search space
for the Viterbi algorithm.

A candidate state is an assignment of the two players to detected
poses in a given frame.

A state is represented as:
(player_0_detection_idx, player_1_detection_idx)

where each value is either:
- an index into FrameDetections.person_detections
- the missing detection sentinel if that player is not detected

Example states:
(2, 0)      player_0 is detection 2, player_1 is detection 0
(2, -1)     player_0 is detection 2, player_1 is missing
(-1, 3)     player_0 is missing, player_1 is detection 3
(-1, -1)    both players are missing
"""


from ai_judo_coach.inference.inference_schemas import ClipDetections, FrameDetections
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


CandidateState = tuple[int, int]


def generate_candidate_states_for_frame(
    frame_detections: FrameDetections,
    config: PlayerDetectionConfig,
) -> list[CandidateState]:
    """
    Generates all possible candidate player-assignment states
    for a single frame.
    """

    sentinel = config.missing_detection_sentinel

    # not frame_detections.person_detections evaluates to True for both None and an empty list
    if not frame_detections.person_detections:
        return [(sentinel, sentinel)]

    num_player_detections = len(frame_detections.person_detections)

    all_candidate_states: list[CandidateState] = []

    # both players assigned to detections
    for player_0_detection_idx in range(num_player_detections):
        for player_1_detection_idx in range(num_player_detections):
            if player_0_detection_idx == player_1_detection_idx:
                continue

            candidate_state = (
                player_0_detection_idx,
                player_1_detection_idx,
            )

            all_candidate_states.append(candidate_state)


    # player_1 missing
    for detection_idx in range(num_player_detections):
        candidate_state = (
            detection_idx,
            sentinel,
        )

        all_candidate_states.append(candidate_state)


    # player_0 missing
    for detection_idx in range(num_player_detections):
        candidate_state = (
            sentinel,
            detection_idx,
        )

        all_candidate_states.append(candidate_state)


    # both players missing
    all_candidate_states.append((sentinel, sentinel))

    return all_candidate_states

    

def generate_candidate_states_for_clip(
    clip_detections: ClipDetections,
    config: PlayerDetectionConfig,
) -> list[list[CandidateState]]:
    """
    Generates candidate player-assignment states
    for every frame in a clip.
    """

    clip_candidate_states = []
    for frame in clip_detections.frame_detections:
        clip_candidate_states.append(
            generate_candidate_states_for_frame(
                frame_detections=frame,
                config=config
            )
        )

    return clip_candidate_states


def is_assignment_missing(
    assignment_idx: int,
    config: PlayerDetectionConfig,
) -> bool:
    """
    Returns True if an assignment index within
    a candidate state tuple takes on the sentinel
    value and hence represents a missing player.
    """

    return assignment_idx == config.missing_detection_sentinel


def state_has_missing_player(
    state: CandidateState,
    config: PlayerDetectionConfig,
) -> bool:
    """
    Returns True if either player is missing in the state.
    """

    player_0_assignment_idx, player_1_assignment_idx = state

    return (
        is_assignment_missing(player_0_assignment_idx, config)
        or is_assignment_missing(player_1_assignment_idx, config)
    )


def state_has_both_players_missing(
    state: CandidateState,
    config: PlayerDetectionConfig,
) -> bool:
    """
    Returns True if both players are missing in the state.
    """

    player_0_assignment_idx, player_1_assignment_idx = state

    return (
        is_assignment_missing(player_0_assignment_idx, config)
        and is_assignment_missing(player_1_assignment_idx, config)
    )


def state_has_duplicate_detection(
    state: CandidateState,
    config: PlayerDetectionConfig,
) -> bool:
    """
    Returns True if both players are assigned to the same
    non-missing detection.
    """

    player_0_assignment_idx, player_1_assignment_idx = state

    if state_has_missing_player(state, config):
        return False

    return player_0_assignment_idx == player_1_assignment_idx
