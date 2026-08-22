"""
This file contains functions for computing
heuristic scores for pairs of detections.
"""

import math

from ai_judo_coach.inference.inference_schemas import PersonDetection
from ai_judo_coach.inference.player_detection.tracking_config import PlayerDetectionConfig
from ai_judo_coach.inference.player_detection.features import (
    average_keypoint_proximity,
    bbox_iou,
    normalized_distance_between_bbox_centers,
)


def pair_score(
    person_detection_a: PersonDetection,
    person_detection_b: PersonDetection,
    config: PlayerDetectionConfig,
) -> float:
    """
    Computes a heuristic score on the interval [0.0, 1.0]
    representing how plausible two detections are as the
    pair of interacting players.

    This score is identity agnostic: swapping detection A
    and detection B should give the same score.

    Higher values indicate a more plausible player pair.
    """

    bbox_overlap_score = bbox_iou(
        person_detection_a=person_detection_a, 
        person_detection_b=person_detection_b
    )

    average_keypoint_proximity_score = average_keypoint_proximity(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=config.keypoint_confidence_threshold
    )

    bbox_center_closeness_score = _bbox_center_distance_to_closeness_score(
        normalized_bbox_center_distance=normalized_distance_between_bbox_centers(
            person_detection_a=person_detection_a,
            person_detection_b=person_detection_b
        )
    )

    weighted_sum = (
        bbox_overlap_score * config.bbox_overlap_weight
        + average_keypoint_proximity_score * config.average_keypoint_proximity_weight
        + bbox_center_closeness_score * config.pair_bbox_center_closeness_weight
    )

    sum_of_weight_values = (
        config.bbox_overlap_weight
        + config.average_keypoint_proximity_weight
        + config.pair_bbox_center_closeness_weight
    )

    if sum_of_weight_values <= 0.0:
        return 0.0

    normalized_weighted_sum = weighted_sum / sum_of_weight_values

    return float(normalized_weighted_sum)



def _bbox_center_distance_to_closeness_score(
    normalized_bbox_center_distance: float,
) -> float:
    """
    Converts normalized bbox center distance into
    a closeness score on the interval [0.0, 1.0].
    """

    max_possible_normalized_distance = math.sqrt(2.0)

    normalized_distance = (
        normalized_bbox_center_distance
        / max_possible_normalized_distance
    )

    closeness_score = 1.0 - normalized_distance

    closeness_score = max(0.0, closeness_score)
    closeness_score = min(1.0, closeness_score)

    return float(closeness_score)
