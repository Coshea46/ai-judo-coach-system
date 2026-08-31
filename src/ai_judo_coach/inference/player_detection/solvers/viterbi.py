import math

import numpy as np

from ai_judo_coach.inference.player_detection.scoring import (
    build_transition_score_matrix,
    state_score,
)
from ai_judo_coach.inference.player_detection.candidate_states import (
    CandidateState,
)
from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
)
from ai_judo_coach.inference.player_detection.tracking_config import (
    PlayerDetectionConfig,
)


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

        frame_mlp_table_row: dict[
            CandidateState,
            float,
        ] = {}
        frame_predecessor_table_row: dict[
            CandidateState,
            CandidateState | None,
        ] = {}

        current_frame_detections = (
            clip_detections.frame_detections[frame_idx]
        )

        detection_score_cache: dict[
            int,
            float,
        ] = {}

        pair_score_cache: dict[
            tuple[int, int],
            float,
        ] = {}

        current_state_scores = np.asarray(
            [
                state_score(
                    state=candidate_state,
                    frame_detections=(
                        current_frame_detections
                    ),
                    config=config,
                    detection_score_cache=(
                        detection_score_cache
                    ),
                    pair_score_cache=pair_score_cache,
                )
                for candidate_state
                in frame_candidate_states
            ],
            dtype=np.float64,
        )

        if frame_idx == 0:
            for (
                candidate_state,
                current_state_score,
            ) in zip(
                frame_candidate_states,
                current_state_scores,
                strict=True,
            ):
                frame_mlp_table_row[candidate_state] = float(
                    VITERBI_PRIOR_SCORE
                    + current_state_score
                )

                # predecessor should be None for entire 0th row in table
                frame_predecessor_table_row[candidate_state] = None

        else:
            previous_frame_detections = (
                clip_detections.frame_detections[
                    frame_idx - 1
                ]
            )

            previous_states = list(
                mlp_table[frame_idx - 1]
            )
            previous_mlp_values = np.asarray(
                [
                    mlp_table[frame_idx - 1][state]
                    for state in previous_states
                ],
                dtype=np.float64,
            )

            transition_score_matrix = (
                build_transition_score_matrix(
                    previous_states=previous_states,
                    current_states=(
                        frame_candidate_states
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

            candidate_mlp_values = (
                previous_mlp_values[:, np.newaxis]
                + transition_score_matrix
                + current_state_scores[np.newaxis, :]
            )

            selectable_predecessors = (
                candidate_mlp_values
                > -math.inf
            )
            has_selectable_predecessor = np.any(
                selectable_predecessors,
                axis=0,
            )

            if not np.all(
                has_selectable_predecessor
            ):
                invalid_state_index = int(
                    np.flatnonzero(
                        ~has_selectable_predecessor
                    )[0]
                )
                invalid_state = frame_candidate_states[
                    invalid_state_index
                ]

                raise RuntimeError(
                    "Could not find a predecessor for "
                    f"state {invalid_state} in frame "
                    f"{frame_idx}."
                )

            selectable_mlp_values = np.where(
                selectable_predecessors,
                candidate_mlp_values,
                -math.inf,
            )

            best_predecessor_indices = np.argmax(
                selectable_mlp_values,
                axis=0,
            )

            for (
                current_state_index,
                candidate_state,
            ) in enumerate(
                frame_candidate_states
            ):
                best_predecessor_index = int(
                    best_predecessor_indices[
                        current_state_index
                    ]
                )

                frame_mlp_table_row[candidate_state] = float(
                    candidate_mlp_values[
                        best_predecessor_index,
                        current_state_index,
                    ]
                )

                frame_predecessor_table_row[candidate_state] = (
                    previous_states[
                        best_predecessor_index
                    ]
                )

        mlp_table.append(frame_mlp_table_row)
        predecessor_table.append(
            frame_predecessor_table_row
        )

    # find the highest-scoring state in the final frame
    mlp_end_state = max(
        mlp_table[num_frames - 1],
        key=lambda state: (
            mlp_table[num_frames - 1][state]
        ),
    )

    # walk backwards through the predecessor table
    output_path: list[CandidateState] = [
        mlp_end_state
    ]
    current_state = mlp_end_state

    for frame_idx in range(
        num_frames - 1,
        0,
        -1,
    ):
        predecessor_state = (
            predecessor_table[frame_idx][current_state]
        )

        if predecessor_state is None:
            raise RuntimeError(
                "Encountered a missing predecessor while "
                f"backtracking from frame {frame_idx}."
            )

        output_path.append(predecessor_state)
        current_state = predecessor_state

    output_path.reverse()

    return output_path
