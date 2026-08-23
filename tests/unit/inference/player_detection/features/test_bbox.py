import math

import numpy as np
import pytest

from ai_judo_coach.inference.inference_schemas import (
    PersonDetection,
)
from ai_judo_coach.inference.player_detection.features.bbox import (
    bbox_area,
    bbox_center,
    bbox_height,
    bbox_width,
    normalized_bbox_area,
    normalized_bbox_center,
    normalized_bbox_distance_to_frame_center,
    normalized_bbox_height,
    normalized_bbox_width,
)


def _create_person_detection(
    bbox_xyxy_px: np.ndarray | list[float] = (
        10.0,
        20.0,
        50.0,
        80.0,
    ),
    bbox_xyxy_normalized: np.ndarray | list[float] = (
        0.1,
        0.2,
        0.5,
        0.8,
    ),
) -> PersonDetection:
    """Create one person detection for bounding-box tests."""

    return PersonDetection(
        detection_idx=0,
        track_id=1,
        bbox_xyxy_px=np.asarray(
            bbox_xyxy_px,
            dtype=np.float32,
        ),
        bbox_xyxy_normalized=np.asarray(
            bbox_xyxy_normalized,
            dtype=np.float32,
        ),
        bbox_conf=0.9,
        keypoints_xy_px=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_xy_norm=np.zeros(
            (17, 2),
            dtype=np.float32,
        ),
        keypoints_conf=np.ones(
            17,
            dtype=np.float32,
        ),
    )


def test_bbox_width_returns_pixel_width() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_px=[
            10.0,
            20.0,
            50.0,
            80.0,
        ],
    )

    result = bbox_width(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(40.0)


def test_bbox_height_returns_pixel_height() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_px=[
            10.0,
            20.0,
            50.0,
            80.0,
        ],
    )

    result = bbox_height(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(60.0)


@pytest.mark.parametrize(
    (
        "bbox_coordinates",
        "expected_center",
    ),
    [
        (
            [10.0, 20.0, 50.0, 80.0],
            (30.0, 50.0),
        ),
        (
            [-20.0, -10.0, 20.0, 30.0],
            (0.0, 10.0),
        ),
        (
            [5.0, 7.0, 5.0, 7.0],
            (5.0, 7.0),
        ),
    ],
)
def test_bbox_center_returns_pixel_centre(
    bbox_coordinates: list[float],
    expected_center: tuple[float, float],
) -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_px=bbox_coordinates,
    )

    result = bbox_center(
        person_detection=person_detection
    )

    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert result == pytest.approx(
        expected_center
    )


def test_bbox_area_returns_pixel_area() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_px=[
            10.0,
            20.0,
            50.0,
            80.0,
        ],
    )

    result = bbox_area(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(2400.0)


@pytest.mark.parametrize(
    "bbox_coordinates",
    [
        [10.0, 20.0, 10.0, 80.0],
        [10.0, 20.0, 50.0, 20.0],
        [10.0, 20.0, 10.0, 20.0],
    ],
)
def test_bbox_area_returns_zero_for_degenerate_box(
    bbox_coordinates: list[float],
) -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_px=bbox_coordinates,
    )

    result = bbox_area(
        person_detection=person_detection
    )

    assert result == pytest.approx(0.0)


def test_normalized_bbox_width_returns_normalized_width() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=[
            0.1,
            0.2,
            0.5,
            0.8,
        ],
    )

    result = normalized_bbox_width(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.4)


def test_normalized_bbox_height_returns_normalized_height() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=[
            0.1,
            0.2,
            0.5,
            0.8,
        ],
    )

    result = normalized_bbox_height(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.6)


@pytest.mark.parametrize(
    (
        "bbox_coordinates",
        "expected_center",
    ),
    [
        (
            [0.1, 0.2, 0.5, 0.8],
            (0.3, 0.5),
        ),
        (
            [0.0, 0.0, 1.0, 1.0],
            (0.5, 0.5),
        ),
        (
            [0.2, 0.4, 0.2, 0.4],
            (0.2, 0.4),
        ),
    ],
)
def test_normalized_bbox_center_returns_normalized_centre(
    bbox_coordinates: list[float],
    expected_center: tuple[float, float],
) -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=(
            bbox_coordinates
        ),
    )

    result = normalized_bbox_center(
        person_detection=person_detection
    )

    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert result == pytest.approx(
        expected_center
    )


def test_normalized_bbox_area_returns_normalized_area() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=[
            0.1,
            0.2,
            0.5,
            0.8,
        ],
    )

    result = normalized_bbox_area(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.24)


@pytest.mark.parametrize(
    (
        "bbox_coordinates",
        "expected_width",
        "expected_height",
    ),
    [
        (
            [0.8, 0.2, 0.4, 0.7],
            0.0,
            0.5,
        ),
        (
            [0.1, 0.9, 0.6, 0.3],
            0.5,
            0.0,
        ),
        (
            [0.8, 0.9, 0.4, 0.3],
            0.0,
            0.0,
        ),
    ],
)
def test_normalized_bbox_dimensions_do_not_become_negative(
    bbox_coordinates: list[float],
    expected_width: float,
    expected_height: float,
) -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=(
            bbox_coordinates
        ),
    )

    assert normalized_bbox_width(
        person_detection=person_detection
    ) == pytest.approx(expected_width)

    assert normalized_bbox_height(
        person_detection=person_detection
    ) == pytest.approx(expected_height)

    assert normalized_bbox_area(
        person_detection=person_detection
    ) == pytest.approx(
        expected_width * expected_height
    )


def test_normalized_bbox_distance_is_zero_at_frame_center() -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=[
            0.4,
            0.3,
            0.6,
            0.7,
        ],
    )

    result = normalized_bbox_distance_to_frame_center(
        person_detection=person_detection
    )

    assert isinstance(result, float)
    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    (
        "bbox_coordinates",
        "expected_distance",
    ),
    [
        (
            [0.0, 0.0, 0.0, 0.0],
            math.sqrt(0.5),
        ),
        (
            [1.0, 1.0, 1.0, 1.0],
            math.sqrt(0.5),
        ),
        (
            [0.0, 0.0, 0.5, 0.5],
            math.sqrt(0.125),
        ),
        (
            [0.5, 0.5, 1.0, 1.0],
            math.sqrt(0.125),
        ),
    ],
)
def test_normalized_bbox_distance_to_frame_center(
    bbox_coordinates: list[float],
    expected_distance: float,
) -> None:
    person_detection = _create_person_detection(
        bbox_xyxy_normalized=(
            bbox_coordinates
        ),
    )

    result = normalized_bbox_distance_to_frame_center(
        person_detection=person_detection
    )

    assert result == pytest.approx(
        expected_distance
    )
