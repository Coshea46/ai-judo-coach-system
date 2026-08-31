"""
This file contains utility functions for 
quantifying interactions between 2 given poses
"""

import math

import numpy as np

from ai_judo_coach.inference.inference_schemas import PersonDetection
from .bbox import(
    bbox_area,
    bbox_center,
    normalized_bbox_center
)


def bbox_iou(
    person_detection_a: PersonDetection, 
    person_detection_b: PersonDetection
) -> float:
    """
    Computes the intersection over union for
    two pose bounding boxes
    

    Note: internal variable names in this function
    use lower_y / upper_y terminology as if coords
    were being read in a standard bottom-left-origin
    coordinate system, even though YOLO output coords
    are with respect to the top left of the screen.
    This naming convention was chosen for readability
    purposes and makes no functional difference
    to the code.

    """

    # unbox coords for player a
    bbox_a_left_x = person_detection_a.bbox_xyxy_px[0]
    bbox_a_right_x = person_detection_a.bbox_xyxy_px[2]
    bbox_a_lower_y = person_detection_a.bbox_xyxy_px[1]
    bbox_a_upper_y = person_detection_a.bbox_xyxy_px[3]

    # unbox coords for player b
    bbox_b_left_x = person_detection_b.bbox_xyxy_px[0]
    bbox_b_right_x = person_detection_b.bbox_xyxy_px[2]
    bbox_b_lower_y = person_detection_b.bbox_xyxy_px[1]
    bbox_b_upper_y = person_detection_b.bbox_xyxy_px[3]

    # compute coords for rectangle of overlap
    intersection_left_x = max(bbox_a_left_x, bbox_b_left_x)
    intersection_right_x = min(bbox_a_right_x, bbox_b_right_x)
    intersection_lower_y = max(bbox_a_lower_y, bbox_b_lower_y)
    intersection_upper_y = min(bbox_a_upper_y, bbox_b_upper_y)

    # compute areas of player boxes
    player_a_bbox_area = bbox_area(person_detection=person_detection_a)
    player_b_bbox_area = bbox_area(person_detection=person_detection_b)

    # compute area of intersection
    intersection_box_width = max(
        0.0,
        intersection_right_x - intersection_left_x
    )
    intersection_box_height = max(
        0.0,
        intersection_upper_y - intersection_lower_y
    )
    intersection_box_area = intersection_box_width * intersection_box_height

    # now compute IoU
    union_area = (
        player_a_bbox_area
        + player_b_bbox_area
        - intersection_box_area
    )

    if union_area <= 0.0:
        return 0.0

    iou = intersection_box_area / union_area

    return float(iou)



def distance_between_bbox_centers(
    person_detection_a: PersonDetection,
    person_detection_b: PersonDetection,
) -> float:
    """
    Computes the distance in pixels between
    the centers of two pose bounding boxes.
    """

    detection_a_bbox_center = bbox_center(
        person_detection=person_detection_a
    )

    detection_b_bbox_center = bbox_center(
        person_detection=person_detection_b
    )

    return float(math.dist(detection_a_bbox_center, detection_b_bbox_center))



def normalized_distance_between_bbox_centers(
    person_detection_a: PersonDetection,
    person_detection_b: PersonDetection,
) -> float:
    """
    Computes the distance between the normalized
    centers of two pose bounding boxes.
    """

    norm_detection_a_center = normalized_bbox_center(
        person_detection=person_detection_a
    )

    norm_detection_b_center = normalized_bbox_center(
        person_detection=person_detection_b
    )

    return float(math.dist(norm_detection_a_center, norm_detection_b_center))



# keep public for now in case want to use for diagnostics later
def average_nearest_keypoint_distance(
    person_detection_a: PersonDetection,
    person_detection_b: PersonDetection,
    min_keypoint_confidence: float,
) -> float | None:
    """
    Computes the average nearest-keypoint distance
    between two poses using normalized keypoint coords.

    Can think of the returned float as representing
    the general distance between the 2 poses

    Returns None if either pose has no usable keypoints.
    """

    detection_a_visible_keypoints = _visible_normalized_keypoints(
        person_detection_a,
        min_keypoint_confidence=min_keypoint_confidence
    )

    detection_b_visible_keypoints = _visible_normalized_keypoints(
        person_detection_b,
        min_keypoint_confidence=min_keypoint_confidence
    )

    if (
        len(detection_a_visible_keypoints) == 0
        or len(detection_b_visible_keypoints) == 0
    ):
        return None

    detection_a_visible_keypoints = (
        detection_a_visible_keypoints.astype(
            np.float64,
            copy=False,
        )
    )
    detection_b_visible_keypoints = (
        detection_b_visible_keypoints.astype(
            np.float64,
            copy=False,
        )
    )

    keypoint_differences = (
        detection_a_visible_keypoints[:, np.newaxis, :]
        - detection_b_visible_keypoints[np.newaxis, :, :]
    )

    pairwise_keypoint_distances = np.linalg.norm(
        keypoint_differences,
        axis=2,
    )

    detection_a_avg_kp_distance = float(
        np.min(
            pairwise_keypoint_distances,
            axis=1,
        ).mean()
    )

    detection_b_avg_kp_distance = float(
        np.min(
            pairwise_keypoint_distances,
            axis=0,
        ).mean()
    )

    return float(
        (
            detection_a_avg_kp_distance
            + detection_b_avg_kp_distance
        )
        / 2
    )




def average_keypoint_proximity(
    person_detection_a: PersonDetection,
    person_detection_b: PersonDetection,
    min_keypoint_confidence: float,
) -> float:
    """
    Computes a proximity score on the interval [0.0, 1.0]
    based on average nearest-keypoint distance.

    Normalizes score with respect to max normalized
    keypoint distance within a frame (which is sqrt(2)
    since corner to corner gives distance sqrt(2) in a 
    1.0 by 1.0 grid).

    Higher values mean the two poses are closer.
    """

    # can be encoded like this since frame dimensions used are normalized
    max_possible_normalized_distance = math.sqrt(2.0)

    average_pose_distance = average_nearest_keypoint_distance(
        person_detection_a=person_detection_a,
        person_detection_b=person_detection_b,
        min_keypoint_confidence=min_keypoint_confidence
    )

    if average_pose_distance is None:
        return 0.0

    normalized_distance = (
        average_pose_distance
        / max_possible_normalized_distance
    )

    # need subtraction since want poses that are closer together to have a higher score
    proximity = 1.0 - normalized_distance

    proximity = max(0.0, proximity)
    proximity = min(1.0, proximity)

    return float(proximity)



def _visible_normalized_keypoints(
    person_detection: PersonDetection,
    min_keypoint_confidence: float,
) -> np.ndarray:
    """
    Returns values of the visible normalized 
    keypoints for a pose.
    Does not return any information on which
    keypoints on the pose they are, just their
    coordinate values.
    e.g. does return [[0.42, 0.87], [0.20, 0.83]],
    will not return e.g. ["right hip", "left elbow"]

    For a keypoint to be classed as visible it must meet all
    of these criteria:
    - confidence score >= min_keypoint_confidence
    - x,y values are finite
    - at least one of the x,y values is non-zero
    """

    normalized_keypoints = person_detection.keypoints_xy_norm
    keypoint_confidence_scores = person_detection.keypoints_conf

    confident_keypoints_mask = (
        keypoint_confidence_scores >= min_keypoint_confidence
    )

    finite_keypoints_mask = np.all(
        np.isfinite(normalized_keypoints),
        axis=1,
    )

    non_zero_keypoints_mask = ~np.all(
        normalized_keypoints == 0.0,
        axis=1,
    )

    visible_keypoints_mask = (
        confident_keypoints_mask
        & finite_keypoints_mask
        & non_zero_keypoints_mask
    )

    return normalized_keypoints[visible_keypoints_mask]
