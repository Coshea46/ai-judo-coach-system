from dataclasses import dataclass


# Sentinel used in Viterbi states when a player has no assigned detection.
MISSING_DETECTION_SENTINEL = -1


@dataclass(frozen=True, slots=True)
class PlayerDetectionConfig:
    """
    Mainly stores tunable weights and thresholds for the 
    player detection system.
    Also stores sentinel values to be used.

    Weight values are non-negative relative multipliers.
    Feature values should generally be normalized to [0, 1].
    Bonuses are added to scores.
    Penalties are positive magnitudes and are subtracted from scores.

    Storing as class allows easy passing to scoring
    functions.
    """

    missing_detection_sentinel: int = MISSING_DETECTION_SENTINEL

    keypoint_confidence_threshold: float = 0.3

    # bbox score weights
    bbox_confidence_weight: float = 0.2
    bbox_center_closeness_weight: float = 0.4

    # pose score weights and normalization
    mean_keypoint_confidence_weight: float = 0.2
    pose_size_weight: float = 0.6
    max_expected_normalized_body_length: float = 1.0

    # pair score weights (for scoring interactions between poses)
    bbox_overlap_weight: float = 0.4
    pair_bbox_center_closeness_weight: float = 0.5
    average_keypoint_proximity_weight: float = 0.4

    # transition score weights and penalties/bonuses
    same_track_id_bonus: float = 0.4
    bbox_center_distance_penalty_weight: float = 0.5

    # missing state penalties
    one_player_missing_penalty: float = 0.3
    both_players_missing_penalty: float = 0.5

    # interpolation thresholds
    longest_gap_allowed: int = 5

    # pose sequence quality thresholds
    min_resolved_keypoints_per_usable_frame: int = 6
    max_unusable_frame_fraction_per_player: float = 0.20
    max_consecutive_unusable_frames_per_player: int = 10
