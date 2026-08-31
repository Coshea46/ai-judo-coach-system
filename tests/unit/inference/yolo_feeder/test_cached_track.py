import numpy as np
import pytest

import ai_judo_coach.inference.yolo_feeder.cached_track as cached_track
from ai_judo_coach.inference.inference_schemas import (
    FrameDetections,
    PersonDetection,
)


FRAME_SHAPE_HW = (100, 200)


def _create_person_detection(
    detection_idx: int,
    marker: float,
    bbox_xyxy_px: list[float],
    bbox_conf: float = 0.9,
    track_id: int | None = None,
) -> PersonDetection:
    """Create a valid person detection with identifiable keypoints."""

    bbox_array = np.asarray(
        bbox_xyxy_px,
        dtype=np.float32,
    )

    bbox_normalisation = np.asarray(
        [
            FRAME_SHAPE_HW[1],
            FRAME_SHAPE_HW[0],
            FRAME_SHAPE_HW[1],
            FRAME_SHAPE_HW[0],
        ],
        dtype=np.float32,
    )

    keypoints_xy_px = np.full(
        (17, 2),
        marker,
        dtype=np.float32,
    )

    keypoint_normalisation = np.asarray(
        [
            FRAME_SHAPE_HW[1],
            FRAME_SHAPE_HW[0],
        ],
        dtype=np.float32,
    )

    return PersonDetection(
        detection_idx=detection_idx,
        track_id=track_id,
        bbox_xyxy_px=bbox_array,
        bbox_xyxy_normalized=(
            bbox_array / bbox_normalisation
        ),
        bbox_conf=bbox_conf,
        keypoints_xy_px=keypoints_xy_px,
        keypoints_xy_norm=(
            keypoints_xy_px / keypoint_normalisation
        ),
        keypoints_conf=np.full(
            (17,),
            marker / 10.0,
            dtype=np.float32,
        ),
    )


def _create_frame_detections(
    frame_idx: int,
    person_detections: list[PersonDetection],
) -> FrameDetections:
    """Create frame detections using the shared test frame shape."""

    return FrameDetections(
        person_detections=person_detections,
        frame_idx=frame_idx,
        frame_shape_hw=FRAME_SHAPE_HW,
    )


def test_collect_cached_tracked_clip_detections_rejects_mismatched_lengths(
    mocker,
) -> None:
    yolo_model = mocker.Mock()

    create_tracker_mock = mocker.patch.object(
        cached_track,
        "_create_tracker",
    )

    with pytest.raises(
        ValueError,
        match=(
            "clip_as_numpy and absolute_frame_indices "
            "must have the same length"
        ),
    ):
        cached_track.collect_cached_tracked_clip_detections(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[
                np.zeros(
                    (100, 200, 3),
                    dtype=np.uint8,
                ),
            ],
            absolute_frame_indices=[],
            compute_device="cpu",
            pose_detection_cache={},
            clip_id="clip_0",
        )

    yolo_model.predict.assert_not_called()
    create_tracker_mock.assert_not_called()


def test_collect_cached_tracked_clip_detections_reuses_overlapping_frames(
    mocker,
) -> None:
    frames = [
        np.zeros(
            (100, 200, 3),
            dtype=np.uint8,
        ),
        np.ones(
            (100, 200, 3),
            dtype=np.uint8,
        ),
        np.full(
            (100, 200, 3),
            2,
            dtype=np.uint8,
        ),
    ]

    yolo_model = mocker.Mock()
    yolo_model.predict.return_value = [
        mocker.Mock(),
    ]

    def adapt_result(
        yolo_frame_result,
        frame_idx: int,
    ) -> FrameDetections:
        del yolo_frame_result

        return _create_frame_detections(
            frame_idx=frame_idx,
            person_detections=[
                _create_person_detection(
                    detection_idx=0,
                    marker=float(frame_idx),
                    bbox_xyxy_px=[
                        10.0,
                        20.0,
                        60.0,
                        80.0,
                    ],
                ),
            ],
        )

    result_adapter_mock = mocker.patch.object(
        cached_track,
        "result_to_frame_detections",
        side_effect=adapt_result,
    )

    first_tracker = mocker.Mock()
    second_tracker = mocker.Mock()

    first_tracker.update.return_value = np.empty(
        (0, 8),
        dtype=np.float32,
    )
    second_tracker.update.return_value = np.empty(
        (0, 8),
        dtype=np.float32,
    )

    create_tracker_mock = mocker.patch.object(
        cached_track,
        "_create_tracker",
        side_effect=[
            first_tracker,
            second_tracker,
        ],
    )

    pose_detection_cache: dict[int, FrameDetections] = {}

    first_result = (
        cached_track
        .collect_cached_tracked_clip_detections(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[
                frames[0],
                frames[1],
            ],
            absolute_frame_indices=[
                10,
                11,
            ],
            compute_device="cuda:0",
            pose_detection_cache=(
                pose_detection_cache
            ),
            clip_id="clip_0",
        )
    )

    second_result = (
        cached_track
        .collect_cached_tracked_clip_detections(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[
                frames[1],
                frames[2],
            ],
            absolute_frame_indices=[
                11,
                12,
            ],
            compute_device="cuda:0",
            pose_detection_cache=(
                pose_detection_cache
            ),
            clip_id="clip_1",
        )
    )

    assert yolo_model.predict.call_count == 3

    predicted_frames = [
        prediction_call.kwargs["source"]
        for prediction_call
        in yolo_model.predict.call_args_list
    ]

    assert predicted_frames[0] is frames[0]
    assert predicted_frames[1] is frames[1]
    assert predicted_frames[2] is frames[2]

    for prediction_call in (
        yolo_model.predict.call_args_list
    ):
        call_arguments = prediction_call.kwargs

        assert call_arguments["conf"] == 0.1
        assert call_arguments["batch"] == 1
        assert call_arguments["device"] == "cuda:0"
        assert call_arguments["verbose"] is False

    assert [
        adapter_call.kwargs["frame_idx"]
        for adapter_call
        in result_adapter_mock.call_args_list
    ] == [
        10,
        11,
        12,
    ]

    assert set(pose_detection_cache) == {
        10,
        11,
        12,
    }

    assert create_tracker_mock.call_count == 2
    assert first_tracker.update.call_count == 2
    assert second_tracker.update.call_count == 2

    assert first_result.clip_id == "clip_0"
    assert second_result.clip_id == "clip_1"

    assert [
        frame.frame_idx
        for frame in first_result.frame_detections
    ] == [
        0,
        1,
    ]

    assert [
        frame.frame_idx
        for frame in second_result.frame_detections
    ] == [
        0,
        1,
    ]


def test_collect_cached_tracked_clip_detections_clears_cached_track_ids(
    mocker,
) -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    incorrectly_tracked_detection = (
        _create_person_detection(
            detection_idx=0,
            marker=1.0,
            bbox_xyxy_px=[
                10.0,
                20.0,
                60.0,
                80.0,
            ],
            track_id=99,
        )
    )

    adapted_frame = _create_frame_detections(
        frame_idx=45,
        person_detections=[
            incorrectly_tracked_detection,
        ],
    )

    yolo_result = mocker.Mock()
    yolo_model = mocker.Mock()
    yolo_model.predict.return_value = [
        yolo_result,
    ]

    result_adapter_mock = mocker.patch.object(
        cached_track,
        "result_to_frame_detections",
        return_value=adapted_frame,
    )

    tracker = mocker.Mock()
    tracker.update.return_value = np.empty(
        (0, 8),
        dtype=np.float32,
    )

    create_tracker_mock = mocker.patch.object(
        cached_track,
        "_create_tracker",
        return_value=tracker,
    )

    pose_detection_cache: dict[int, FrameDetections] = {}

    result = (
        cached_track
        .collect_cached_tracked_clip_detections(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[frame],
            absolute_frame_indices=[45],
            compute_device="cpu",
            pose_detection_cache=(
                pose_detection_cache
            ),
            clip_id="clip_45",
        )
    )

    yolo_model.predict.assert_called_once_with(
        source=frame,
        conf=0.1,
        batch=1,
        device="cpu",
        verbose=False,
    )

    result_adapter_mock.assert_called_once_with(
        yolo_frame_result=yolo_result,
        frame_idx=45,
    )

    create_tracker_mock.assert_called_once_with(
        tracker_path="bytetrack.yaml",
    )

    assert (
        pose_detection_cache[45]
        .person_detections[0]
        .track_id
        is None
    )

    assert (
        result.frame_detections[0]
        .person_detections[0]
        .track_id
        is None
    )


def test_collect_cached_tracked_clip_detections_uses_tracker_source_indices(
    mocker,
) -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    first_source_detection = (
        _create_person_detection(
            detection_idx=0,
            marker=1.0,
            bbox_xyxy_px=[
                20.0,
                10.0,
                60.0,
                50.0,
            ],
            bbox_conf=0.91,
        )
    )

    second_source_detection = (
        _create_person_detection(
            detection_idx=1,
            marker=2.0,
            bbox_xyxy_px=[
                70.0,
                20.0,
                120.0,
                80.0,
            ],
            bbox_conf=0.82,
        )
    )

    cached_frame = _create_frame_detections(
        frame_idx=55,
        person_detections=[
            first_source_detection,
            second_source_detection,
        ],
    )

    tracker = mocker.Mock()
    tracker.update.return_value = np.asarray(
        [
            [
                -10.0,
                20.0,
                220.0,
                120.0,
                7.0,
                0.80,
                0.0,
                1.0,
            ],
            [
                21.0,
                11.0,
                61.0,
                51.0,
                8.0,
                0.70,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float32,
    )

    mocker.patch.object(
        cached_track,
        "_create_tracker",
        return_value=tracker,
    )

    yolo_model = mocker.Mock()

    result = (
        cached_track
        .collect_cached_tracked_clip_detections(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[frame],
            absolute_frame_indices=[55],
            compute_device="cpu",
            pose_detection_cache={
                55: cached_frame,
            },
            clip_id="clip_55",
        )
    )

    yolo_model.predict.assert_not_called()
    tracker.update.assert_called_once()

    tracker_input, supplied_frame = (
        tracker.update.call_args.args
    )

    assert supplied_frame is frame

    np.testing.assert_allclose(
        tracker_input.data,
        np.asarray(
            [
                [
                    20.0,
                    10.0,
                    60.0,
                    50.0,
                    0.91,
                    0.0,
                ],
                [
                    70.0,
                    20.0,
                    120.0,
                    80.0,
                    0.82,
                    0.0,
                ],
            ],
            dtype=np.float32,
        ),
    )

    output_frame = result.frame_detections[0]

    assert output_frame.frame_idx == 0
    assert output_frame.frame_shape_hw == (
        100,
        200,
    )

    first_output = output_frame.person_detections[0]
    second_output = output_frame.person_detections[1]

    assert first_output.detection_idx == 0
    assert first_output.track_id == 7
    assert first_output.bbox_conf == pytest.approx(
        0.80
    )

    np.testing.assert_allclose(
        first_output.bbox_xyxy_px,
        np.asarray(
            [
                0.0,
                20.0,
                200.0,
                100.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        first_output.bbox_xyxy_normalized,
        np.asarray(
            [
                0.0,
                0.2,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        first_output.keypoints_xy_px,
        second_source_detection.keypoints_xy_px,
    )

    np.testing.assert_array_equal(
        first_output.keypoints_conf,
        second_source_detection.keypoints_conf,
    )

    assert second_output.detection_idx == 1
    assert second_output.track_id == 8
    assert second_output.bbox_conf == pytest.approx(
        0.70
    )

    np.testing.assert_allclose(
        second_output.bbox_xyxy_normalized,
        np.asarray(
            [
                0.105,
                0.11,
                0.305,
                0.51,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        second_output.keypoints_xy_px,
        first_source_detection.keypoints_xy_px,
    )

    # Window-specific tracking must not mutate cached detections.
    assert first_source_detection.track_id is None
    assert second_source_detection.track_id is None


def test_empty_tracker_output_copies_untracked_detections(
    mocker,
) -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    cached_detection = _create_person_detection(
        detection_idx=4,
        marker=3.0,
        bbox_xyxy_px=[
            30.0,
            20.0,
            90.0,
            70.0,
        ],
    )

    cached_frame = _create_frame_detections(
        frame_idx=75,
        person_detections=[
            cached_detection,
        ],
    )

    tracker = mocker.Mock()
    tracker.update.return_value = np.empty(
        (0, 8),
        dtype=np.float32,
    )

    mocker.patch.object(
        cached_track,
        "_create_tracker",
        return_value=tracker,
    )

    result = (
        cached_track
        .collect_cached_tracked_clip_detections(
            yolo_model=mocker.Mock(),
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[frame],
            absolute_frame_indices=[75],
            compute_device="cpu",
            pose_detection_cache={
                75: cached_frame,
            },
            clip_id="clip_75",
        )
    )

    output_detection = (
        result.frame_detections[0]
        .person_detections[0]
    )

    assert output_detection is not cached_detection
    assert output_detection.detection_idx == 0
    assert output_detection.track_id is None

    np.testing.assert_array_equal(
        output_detection.bbox_xyxy_px,
        cached_detection.bbox_xyxy_px,
    )
    np.testing.assert_array_equal(
        output_detection.keypoints_xy_px,
        cached_detection.keypoints_xy_px,
    )

    assert not np.shares_memory(
        output_detection.bbox_xyxy_px,
        cached_detection.bbox_xyxy_px,
    )
    assert not np.shares_memory(
        output_detection.keypoints_xy_px,
        cached_detection.keypoints_xy_px,
    )


def test_cache_population_requires_one_result_per_frame(
    mocker,
) -> None:
    yolo_model = mocker.Mock()
    yolo_model.predict.return_value = []

    create_tracker_mock = mocker.patch.object(
        cached_track,
        "_create_tracker",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Expected one YOLO result for one input frame, "
            "received 0"
        ),
    ):
        cached_track.collect_cached_tracked_clip_detections(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=[
                np.zeros(
                    (100, 200, 3),
                    dtype=np.uint8,
                ),
            ],
            absolute_frame_indices=[0],
            compute_device="cpu",
            pose_detection_cache={},
            clip_id="clip_0",
        )

    create_tracker_mock.assert_not_called()
