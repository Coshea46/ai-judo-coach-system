"""
This file contains functions for computing
heuristic transition scores between candidate
player-assignment states in adjacent frames.

A transition score rewards temporal consistency
and penalises unlikely identity/movement changes.

It uses:
- ByteTrack ID consistency
- bbox centre movement between frames

Transitions involving a missing assignment make
no contribution for that player.

Higher values indicate more plausible transitions.
"""

import math

import numpy as np

from ai_judo_coach.inference.inference_schemas import (
    FrameDetections,
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
    is_assignment_missing,
)
from ai_judo_coach.inference.player_detection.features import (
    normalized_distance_between_bbox_centers,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


def transition_score(
    previous_state: CandidateState,
    current_state: CandidateState,
    previous_frame_detections: FrameDetections,
    current_frame_detections: FrameDetections,
    config: PlayerDetectionConfig,
    single_player_score_cache: (
        dict[tuple[int, int], float] | None
    ) = None,
) -> float:
    """
    Computes the transition score between candidate
    player-assignment states in adjacent frames.

    Each player's temporal consistency is scored
    independently before the two scores are summed.
    Need to first score independently so if best
    possible transition is when only one of the players
    is still found, it still ranks higher than if neither
    are found

    Higher values indicate a more plausible transition.
    """

    player_0_transition_score = (
        _get_single_player_transition_score(
            previous_assignment_idx=previous_state[0],
            current_assignment_idx=current_state[0],
            previous_frame_detections=(
                previous_frame_detections
            ),
            current_frame_detections=(
                current_frame_detections
            ),
            config=config,
            score_cache=single_player_score_cache,
        )
    )

    player_1_transition_score = (
        _get_single_player_transition_score(
            previous_assignment_idx=previous_state[1],
            current_assignment_idx=current_state[1],
            previous_frame_detections=(
                previous_frame_detections
            ),
            current_frame_detections=(
                current_frame_detections
            ),
            config=config,
            score_cache=single_player_score_cache,
        )
    )

    return float(
        player_0_transition_score
        + player_1_transition_score
    )


def build_transition_score_matrix(
    previous_states: list[CandidateState],
    current_states: list[CandidateState],
    previous_frame_detections: FrameDetections,
    current_frame_detections: FrameDetections,
    config: PlayerDetectionConfig,
) -> np.ndarray:
    """
    Build transition scores between all supplied candidate states.

    Single-player transitions are calculated once for each distinct
    assignment-index pair. NumPy then combines those scores for the
    two players across all predecessor and current states.

    Rows correspond to previous_states and columns correspond to
    current_states.
    """

    if not previous_states or not current_states:
        return np.empty(
            (
                len(previous_states),
                len(current_states),
            ),
            dtype=np.float64,
        )

    previous_assignment_indices = list(
        dict.fromkeys(
            assignment_idx
            for state in previous_states
            for assignment_idx in state
        )
    )
    current_assignment_indices = list(
        dict.fromkeys(
            assignment_idx
            for state in current_states
            for assignment_idx in state
        )
    )

    previous_assignment_positions = {
        assignment_idx: position
        for position, assignment_idx
        in enumerate(
            previous_assignment_indices
        )
    }
    current_assignment_positions = {
        assignment_idx: position
        for position, assignment_idx
        in enumerate(
            current_assignment_indices
        )
    }

    single_player_scores = np.empty(
        (
            len(previous_assignment_indices),
            len(current_assignment_indices),
        ),
        dtype=np.float64,
    )

    for (
        previous_position,
        previous_assignment_idx,
    ) in enumerate(
        previous_assignment_indices
    ):
        for (
            current_position,
            current_assignment_idx,
        ) in enumerate(
            current_assignment_indices
        ):
            single_player_scores[
                previous_position,
                current_position,
            ] = _single_player_transition_score(
                previous_assignment_idx=(
                    previous_assignment_idx
                ),
                current_assignment_idx=(
                    current_assignment_idx
                ),
                previous_frame_detections=(
                    previous_frame_detections
                ),
                current_frame_detections=(
                    current_frame_detections
                ),
                config=config,
            )

    previous_player_0_positions = np.asarray(
        [
            previous_assignment_positions[state[0]]
            for state in previous_states
        ],
        dtype=np.intp,
    )
    previous_player_1_positions = np.asarray(
        [
            previous_assignment_positions[state[1]]
            for state in previous_states
        ],
        dtype=np.intp,
    )
    current_player_0_positions = np.asarray(
        [
            current_assignment_positions[state[0]]
            for state in current_states
        ],
        dtype=np.intp,
    )
    current_player_1_positions = np.asarray(
        [
            current_assignment_positions[state[1]]
            for state in current_states
        ],
        dtype=np.intp,
    )

    player_0_transition_scores = single_player_scores[
        previous_player_0_positions[:, np.newaxis],
        current_player_0_positions[np.newaxis, :],
    ]
    player_1_transition_scores = single_player_scores[
        previous_player_1_positions[:, np.newaxis],
        current_player_1_positions[np.newaxis, :],
    ]

    return (
        player_0_transition_scores
        + player_1_transition_scores
    )


def _get_single_player_transition_score(
    previous_assignment_idx: int,
    current_assignment_idx: int,
    previous_frame_detections: FrameDetections,
    current_frame_detections: FrameDetections,
    config: PlayerDetectionConfig,
    score_cache: dict[tuple[int, int], float] | None,
) -> float:
    """
    Return one player's transition score, using a caller-supplied cache.

    The cache must only be reused for one fixed pair of adjacent frames.
    """

    if score_cache is None:
        return _single_player_transition_score(
            previous_assignment_idx=(
                previous_assignment_idx
            ),
            current_assignment_idx=(
                current_assignment_idx
            ),
            previous_frame_detections=(
                previous_frame_detections
            ),
            current_frame_detections=(
                current_frame_detections
            ),
            config=config,
        )

    cache_key = (
        previous_assignment_idx,
        current_assignment_idx,
    )

    if cache_key not in score_cache:
        score_cache[cache_key] = (
            _single_player_transition_score(
                previous_assignment_idx=(
                    previous_assignment_idx
                ),
                current_assignment_idx=(
                    current_assignment_idx
                ),
                previous_frame_detections=(
                    previous_frame_detections
                ),
                current_frame_detections=(
                    current_frame_detections
                ),
                config=config,
            )
        )

    return score_cache[cache_key]


def _single_player_transition_score(
    previous_assignment_idx: int,
    current_assignment_idx: int,
    previous_frame_detections: FrameDetections,
    current_frame_detections: FrameDetections,
    config: PlayerDetectionConfig,
) -> float:
    """
    Computes the transition score for one player
    between two adjacent frames.

    Returns 0.0 if either assignment is missing.
    """

    previous_assignment_missing = is_assignment_missing(
        assignment_idx=previous_assignment_idx,
        config=config,
    )

    current_assignment_missing = is_assignment_missing(
        assignment_idx=current_assignment_idx,
        config=config,
    )

    if previous_assignment_missing or current_assignment_missing:
        return 0.0

    previous_detection = previous_frame_detections.person_detections[
        previous_assignment_idx
    ]

    current_detection = current_frame_detections.person_detections[
        current_assignment_idx
    ]

    bbox_center_distance = normalized_distance_between_bbox_centers(
        person_detection_a=previous_detection,
        person_detection_b=current_detection,
    )

    max_possible_normalized_distance = math.sqrt(2.0)

    normalized_bbox_center_distance = (
        bbox_center_distance
        / max_possible_normalized_distance
    )

    normalized_bbox_center_distance = max(
        0.0,
        normalized_bbox_center_distance,
    )

    normalized_bbox_center_distance = min(
        1.0,
        normalized_bbox_center_distance,
    )

    bbox_center_distance_penalty = (
        normalized_bbox_center_distance
        * config.bbox_center_distance_penalty_weight
    )

    player_transition_score = -bbox_center_distance_penalty

    if _same_known_track_id(
        previous_detection=previous_detection,
        current_detection=current_detection,
    ):
        player_transition_score += config.same_track_id_bonus

    return float(player_transition_score)


def _same_known_track_id(
    previous_detection: PersonDetection,
    current_detection: PersonDetection,
) -> bool:
    """
    Returns True if both detections have the same
    known ByteTrack ID.

    Returns False if either track ID is None.
    """

    if (
        previous_detection.track_id is None
        or current_detection.track_id is None
    ):
        return False

    return previous_detection.track_id == current_detection.track_id
