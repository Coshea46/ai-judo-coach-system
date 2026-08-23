import numpy as np
import pytest

from ai_judo_coach.inference.yolo_feeder.track import (
    track_video,
)


def test_track_video_calls_yolo_tracking_with_expected_arguments(
    mocker,
) -> None:
    yolo_model = mocker.Mock()

    clip_as_numpy = [
        np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        ),
        np.ones(
            (720, 1280, 3),
            dtype=np.uint8,
        ),
    ]

    expected_results_stream = iter(
        [
            mocker.Mock(),
            mocker.Mock(),
        ]
    )
    yolo_model.track.return_value = (
        expected_results_stream
    )

    result = track_video(
        yolo_model=yolo_model,
        tracker_path="config/bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device="cuda:0",
    )

    assert result is expected_results_stream

    yolo_model.track.assert_called_once()

    call_arguments = (
        yolo_model.track.call_args.kwargs
    )

    assert (
        call_arguments["source"]
        is clip_as_numpy
    )
    assert call_arguments["stream"] is True
    assert (
        call_arguments["tracker"]
        == "config/bytetrack.yaml"
    )
    assert (
        call_arguments["device"]
        == "cuda:0"
    )
    assert call_arguments["persist"] is False


@pytest.mark.parametrize(
    "compute_device",
    [
        "cpu",
        "cuda:0",
        "mps",
        0,
    ],
)
def test_track_video_passes_compute_device_unchanged(
    mocker,
    compute_device: str | int,
) -> None:
    yolo_model = mocker.Mock()
    expected_results_stream = iter(())

    yolo_model.track.return_value = (
        expected_results_stream
    )

    clip_as_numpy = [
        np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    ]

    result = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device=compute_device,
    )

    assert result is expected_results_stream
    assert (
        yolo_model.track.call_args.kwargs[
            "device"
        ]
        == compute_device
    )


def test_track_video_does_not_consume_results_stream(
    mocker,
) -> None:
    yolo_model = mocker.Mock()

    stream_consumed = False
    expected_result = mocker.Mock()

    def results_stream():
        nonlocal stream_consumed

        stream_consumed = True
        yield expected_result

    expected_results_stream = (
        results_stream()
    )
    yolo_model.track.return_value = (
        expected_results_stream
    )

    result = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=[],
        compute_device="cpu",
    )

    assert result is expected_results_stream
    assert stream_consumed is False

    assert next(result) is expected_result
    assert stream_consumed is True


def test_track_video_disables_tracker_persistence_between_clips(
    mocker,
) -> None:
    yolo_model = mocker.Mock()
    yolo_model.track.return_value = iter(())

    track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=[],
        compute_device="cpu",
    )

    assert (
        yolo_model.track.call_args.kwargs[
            "persist"
        ]
        is False
    )


def test_track_video_propagates_yolo_tracking_failure(
    mocker,
) -> None:
    tracking_error = RuntimeError(
        "YOLO tracking failed"
    )

    yolo_model = mocker.Mock()
    yolo_model.track.side_effect = (
        tracking_error
    )

    clip_as_numpy = [
        np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="YOLO tracking failed",
    ) as exception_info:
        track_video(
            yolo_model=yolo_model,
            tracker_path="bytetrack.yaml",
            clip_as_numpy=clip_as_numpy,
            compute_device="cpu",
        )

    assert (
        exception_info.value
        is tracking_error
    )

    yolo_model.track.assert_called_once()

    call_arguments = (
        yolo_model.track.call_args.kwargs
    )

    assert (
        call_arguments["source"]
        is clip_as_numpy
    )
    assert call_arguments["stream"] is True
    assert (
        call_arguments["tracker"]
        == "bytetrack.yaml"
    )
    assert call_arguments["device"] == "cpu"
    assert call_arguments["persist"] is False
