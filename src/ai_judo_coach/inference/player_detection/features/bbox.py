"""
file contains utility functions for finding descriptive figures,
relating to the bounding box for a single player

The functions can be used in the scoring system
"""
import math

from ai_judo_coach.inference.inference_schemas import PersonDetection


def bbox_width(person_detection: PersonDetection) -> float:
    """
    Computes the width (in pixels) of the bounding box for a given
    PersonDetection pose
    """

    bbox_coords = person_detection.bbox_xyxy_px

    x1 = bbox_coords[0]
    x2 = bbox_coords[2]

    width = max(0.0, x2 - x1)

    return float(width)



def bbox_height(person_detection: PersonDetection) -> float:
    """
    Computes the height (in pixels) of the bounding box for a given
    PersonDetection pose
    """

    bbox_coords = person_detection.bbox_xyxy_px

    y1 = bbox_coords[1]
    y2 = bbox_coords[3]

    height = max(0.0, y2 - y1)

    return float(height)



def bbox_center(
    person_detection: PersonDetection
) -> tuple[float, float]:
    """
    Computes the center x, y coordinate tuple (in pixels) 
    of the bounding box for a given PersonDetection pose
    """

    bbox_coords = person_detection.bbox_xyxy_px

    x1 = bbox_coords[0]
    x2 = bbox_coords[2]
    y1 = bbox_coords[1]
    y2 = bbox_coords[3]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return float(center_x), float(center_y)



def bbox_area(person_detection: PersonDetection) -> float:
    """
    Computes the area in pixels that the bounding
    box for a given PersonDetection pose covers
    """

    width = bbox_width(person_detection)
    height = bbox_height(person_detection)

    return float(width * height)



def normalized_bbox_width(person_detection: PersonDetection) -> float:
    """
    Computes the normalized width (in pixels) of the bounding 
    box for a given PersonDetection pose
    """

    normalized_bbox_coords = person_detection.bbox_xyxy_normalized

    norm_x1 = normalized_bbox_coords[0]
    norm_x2 = normalized_bbox_coords[2]

    norm_width = max(0.0, norm_x2 - norm_x1)

    return float(norm_width)



def normalized_bbox_height(person_detection: PersonDetection) -> float:
    """
    Computes the normalized height (in pixels) of the bounding 
    box for a given PersonDetection pose
    """

    normalized_bbox_coords = person_detection.bbox_xyxy_normalized

    norm_y1 = normalized_bbox_coords[1]
    norm_y2 = normalized_bbox_coords[3]

    norm_height = max(0.0, norm_y2 - norm_y1)

    return float(norm_height)



def normalized_bbox_center(
    person_detection: PersonDetection
) -> tuple[float, float]:
    """
    Computes the normalized center x, y coordinate tuple (in pixels) 
    of the bounding box for a given PersonDetection pose
    """

    normalized_bbox_coords = person_detection.bbox_xyxy_normalized

    norm_x1 = normalized_bbox_coords[0]
    norm_x2 = normalized_bbox_coords[2]
    norm_y1 = normalized_bbox_coords[1]
    norm_y2 = normalized_bbox_coords[3]

    norm_center_x = (norm_x1 + norm_x2) / 2
    norm_center_y = (norm_y1 + norm_y2) / 2

    return float(norm_center_x), float(norm_center_y)



def normalized_bbox_area(person_detection: PersonDetection) -> float:
    """
    Computes the normalized area in pixels that the bounding
    box for a given PersonDetection pose covers
    
    Area is normalized with respect to the frame
    dimensions
    """

    normalized_width = normalized_bbox_width(person_detection)
    normalized_height = normalized_bbox_height(person_detection)

    return float(normalized_width * normalized_height)



def normalized_bbox_distance_to_frame_center(
    person_detection: PersonDetection
) -> float:
    """
    Computes the normalized distance from the 
    center of a pose bounding box to the center
    of the frame

    Area is normalized with respect to the frame
    dimensions
    """

    normalized_frame_center = (0.5, 0.5)

    # store result as tuple, don't unbox since need tuple in dist function
    normalized_bbox_coords = normalized_bbox_center(
        person_detection=person_detection
    )

    return float(math.dist(normalized_bbox_coords, normalized_frame_center))