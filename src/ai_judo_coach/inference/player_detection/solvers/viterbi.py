import math

from ai_judo_coach.inference.player_detection.scoring import (
    state_score,
    transition_score,
)
from ai_judo_coach.inference.player_detection.candidate_states import CandidateState
from ai_judo_coach.inference.inference_schemas import ClipDetections
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


# currently using uniform additive prior
VITERBI_PRIOR_SCORE = 0.0


def viterbi_algorithm(
    candidate_states_by_frame: list[list[CandidateState]],
    clip_detections: ClipDetections,
    config: PlayerDetectionConfig,
) -> list[CandidateState]:
    """
    Finds the highest-scoring sequence of candidate
    player-assignment states across a clip.

    Uses additive heuristic state and transition scores.

    Returns the selected candidate state for each frame,
    ordered from the first frame to the final frame.
    """

    num_frames = len(candidate_states_by_frame)

    if num_frames == 0:
        return []

    if num_frames != len(clip_detections.frame_detections):
        raise ValueError(
            "Number of candidate-state frames must match "
            "number of frames in ClipDetections."
        )

    mlp_table: list[dict[CandidateState, float]] = []
    predecessor_table: list[
        dict[CandidateState, CandidateState | None]
    ] = []

    for frame_idx, frame_candidate_states in enumerate(
        candidate_states_by_frame
    ):
        if len(frame_candidate_states) == 0:
            raise ValueError(
                f"Frame {frame_idx} has no candidate states."
            )

        frame_mlp_table_row: dict[CandidateState, float] = {}
        frame_predecessor_table_row: dict[
            CandidateState,
            CandidateState | None
        ] = {}

        current_frame_detections = (
            clip_detections.frame_detections[frame_idx]
        )

        for candidate_state in frame_candidate_states:
            current_state_score = state_score(
                state=candidate_state,
                frame_detections=current_frame_detections,
                config=config,
            )

            if frame_idx == 0:
                frame_mlp_table_row[candidate_state] = (
                    VITERBI_PRIOR_SCORE
                    + current_state_score
                )

                # predecessor should be None for entire 0th row in table
                frame_predecessor_table_row[candidate_state] = None

            else:
                previous_frame_detections = (
                    clip_detections.frame_detections[frame_idx - 1]
                )

                current_best_mlp_value = -math.inf
                current_best_predecessor: CandidateState | None = None

                # loop through predecessor state mlp values
                for (
                    predecessor_state,
                    predecessor_mlp_value,
                ) in mlp_table[frame_idx - 1].items():
                    candidate_mlp_value = (
                        predecessor_mlp_value
                        + transition_score(
                            previous_state=predecessor_state,
                            current_state=candidate_state,
                            previous_frame_detections=(
                                previous_frame_detections
                            ),
                            current_frame_detections=(
                                current_frame_detections
                            ),
                            config=config,
                        )
                        + current_state_score
                    )

                    if candidate_mlp_value > current_best_mlp_value:
                        current_best_mlp_value = candidate_mlp_value
                        current_best_predecessor = predecessor_state

                if current_best_predecessor is None:
                    raise RuntimeError(
                        "Could not find a predecessor for "
                        f"state {candidate_state} in frame {frame_idx}."
                    )

                frame_mlp_table_row[candidate_state] = (
                    current_best_mlp_value
                )

                frame_predecessor_table_row[candidate_state] = (
                    current_best_predecessor
                )

        mlp_table.append(frame_mlp_table_row)
        predecessor_table.append(frame_predecessor_table_row)

    # find the highest-scoring state in the final frame
    mlp_end_state = max(
        mlp_table[num_frames - 1],
        key=lambda state: mlp_table[num_frames - 1][state],
    )

    # walk backwards through the predecessor table
    output_path: list[CandidateState] = [mlp_end_state]
    current_state = mlp_end_state

    for frame_idx in range(num_frames - 1, 0, -1):
        predecessor_state = predecessor_table[frame_idx][current_state]

        if predecessor_state is None:
            raise RuntimeError(
                "Encountered a missing predecessor while "
                f"backtracking from frame {frame_idx}."
            )

        output_path.append(predecessor_state)
        current_state = predecessor_state

    output_path.reverse()

    return output_path
