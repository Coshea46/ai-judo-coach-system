"""
This file houses the function for computing
the heuristic score for a given detection 
being one of the two judo players in the frame
"""

import math

from ai_judo_coach.inference.inference_schemas import PersonDetection
from ai_judo_coach.inference.player_detection.features import (
    normalized_bbox_distance_to_frame_center,
    mean_keypoint_confidence,
    average_body_length,
)
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig


def detection_score(
    person_detection: PersonDetection,
    config: PlayerDetectionConfig
) -> float:
    """
    Computes a heuristic score on the
    interval [0.0, 1.0] representing
    how plausible a single detection is as
    one of the two players in the clip.

    This score is identity-agnostic: it does
    not distinguish player_0 from player_1.

    Higher values indicate a more plausible
    player detection.
    """

    # bbox related values
    bbox_confidence_score = float(person_detection.bbox_conf)

    bbox_norm_dist_to_center = normalized_bbox_distance_to_frame_center(
        person_detection=person_detection
    )

    bbox_center_closeness_score = _distance_to_center_to_closeness_score(
        normalized_distance_to_center=bbox_norm_dist_to_center
    )

    # keypoint and pose related scores    
    mean_keypoint_confidence_score = mean_keypoint_confidence(
        player_detection=person_detection
    )

    pose_size = average_body_length(
        player_detection=person_detection,
        min_keypoint_confidence=config.keypoint_confidence_threshold
    )

    pose_size_score = _pose_size_to_score(
        pose_size=pose_size,
        config=config
    )

    weighted_sum = (
        bbox_confidence_score * config.bbox_confidence_weight
        + bbox_center_closeness_score * config.bbox_center_closeness_weight
        + mean_keypoint_confidence_score * config.mean_keypoint_confidence_weight
        + pose_size_score * config.pose_size_weight
    )

    sum_of_weight_values = (
        config.bbox_confidence_weight
        + config.bbox_center_closeness_weight
        + config.mean_keypoint_confidence_weight
        + config.pose_size_weight   
    )

    if sum_of_weight_values <= 0.0:
        return 0.0

    normalized_weighted_sum = weighted_sum / sum_of_weight_values

    return float(normalized_weighted_sum)



def _distance_to_center_to_closeness_score(
    normalized_distance_to_center: float,
) -> float:
    """
    Converts normalized bbox distance to frame center
    into a closeness score on interval [0.0, 1.0].
    """

    max_possible_distance_to_center = math.sqrt(0.5)

    normalized_center_distance = (
        normalized_distance_to_center
        / max_possible_distance_to_center
    )

    closeness_score = 1.0 - normalized_center_distance

    closeness_score = max(0.0, closeness_score)
    closeness_score = min(1.0, closeness_score)

    return float(closeness_score)



def _pose_size_to_score(
    pose_size: float,
    config: PlayerDetectionConfig,
) -> float:
    """
    Converts raw normalized body length into a score
    on interval [0.0, 1.0].
    """

    if config.max_expected_normalized_body_length <= 0.0:
        return 0.0

    pose_size_score = (
        pose_size
        / config.max_expected_normalized_body_length
    )

    pose_size_score = max(0.0, pose_size_score)
    pose_size_score = min(1.0, pose_size_score)

    return float(pose_size_score)
