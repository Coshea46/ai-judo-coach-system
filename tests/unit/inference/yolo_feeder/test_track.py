import numpy as np
import pytest

from ai_judo_coach.inference.yolo_feeder.track import (
    track_video,
)


def _create_yolo_model_without_trackers(
    mocker,
):
    """Create a YOLO mock with no existing predictor."""

    yolo_model = mocker.Mock()
    yolo_model.predictor = None

    return yolo_model


def test_track_video_tracks_each_frame_individually(
    mocker,
) -> None:
    yolo_model = (
        _create_yolo_model_without_trackers(
            mocker=mocker,
        )
    )

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

    first_result = mocker.Mock()
    second_result = mocker.Mock()

    yolo_model.track.side_effect = [
        iter([first_result]),
        iter([second_result]),
    ]

    results_stream = track_video(
        yolo_model=yolo_model,
        tracker_path="config/bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device="cuda:0",
    )

    # Tracking is lazy and begins when the returned iterator is consumed.
    yolo_model.track.assert_not_called()

    result = list(results_stream)

    assert result == [
        first_result,
        second_result,
    ]
    assert yolo_model.track.call_count == 2

    for frame_index, track_call in enumerate(
        yolo_model.track.call_args_list
    ):
        call_arguments = track_call.kwargs

        assert (
            call_arguments["source"]
            is clip_as_numpy[frame_index]
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
        assert call_arguments["persist"] is True


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
    yolo_model = (
        _create_yolo_model_without_trackers(
            mocker=mocker,
        )
    )

    yolo_model.track.return_value = iter(())

    clip_as_numpy = [
        np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    ]

    results_stream = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device=compute_device,
    )

    list(results_stream)

    yolo_model.track.assert_called_once()

    call_arguments = (
        yolo_model.track.call_args.kwargs
    )

    assert (
        call_arguments["source"]
        is clip_as_numpy[0]
    )
    assert (
        call_arguments["device"]
        == compute_device
    )
    assert call_arguments["persist"] is True


def test_track_video_does_not_begin_tracking_until_stream_is_consumed(
    mocker,
) -> None:
    yolo_model = (
        _create_yolo_model_without_trackers(
            mocker=mocker,
        )
    )

    stream_consumed = False
    expected_result = mocker.Mock()

    def frame_results_stream():
        nonlocal stream_consumed

        stream_consumed = True
        yield expected_result

    yolo_model.track.return_value = (
        frame_results_stream()
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
        compute_device="cpu",
    )

    assert stream_consumed is False
    yolo_model.track.assert_not_called()

    assert next(result) is expected_result

    assert stream_consumed is True
    yolo_model.track.assert_called_once()


def test_track_video_retains_tracker_state_between_frames(
    mocker,
) -> None:
    yolo_model = (
        _create_yolo_model_without_trackers(
            mocker=mocker,
        )
    )

    clip_as_numpy = [
        np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
        np.ones(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    ]

    yolo_model.track.side_effect = [
        iter([mocker.Mock()]),
        iter([mocker.Mock()]),
    ]

    results_stream = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device="cpu",
    )

    list(results_stream)

    assert yolo_model.track.call_count == 2

    assert all(
        track_call.kwargs["persist"] is True
        for track_call in yolo_model.track.call_args_list
    )


def test_track_video_resets_existing_trackers_before_new_clip(
    mocker,
) -> None:
    yolo_model = mocker.Mock()

    first_tracker = mocker.Mock()
    second_tracker = mocker.Mock()

    yolo_model.predictor.trackers = [
        first_tracker,
        second_tracker,
    ]

    results_stream = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=[],
        compute_device="cpu",
    )

    first_tracker.reset.assert_called_once_with()
    second_tracker.reset.assert_called_once_with()

    assert list(results_stream) == []
    yolo_model.track.assert_not_called()


def test_track_video_handles_predictor_without_trackers(
    mocker,
) -> None:
    yolo_model = mocker.Mock()
    yolo_model.predictor.trackers = None

    results_stream = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=[],
        compute_device="cpu",
    )

    assert list(results_stream) == []
    yolo_model.track.assert_not_called()


def test_track_video_returns_empty_iterator_for_empty_clip(
    mocker,
) -> None:
    yolo_model = (
        _create_yolo_model_without_trackers(
            mocker=mocker,
        )
    )

    result = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=[],
        compute_device="cpu",
    )

    assert list(result) == []
    yolo_model.track.assert_not_called()


def test_track_video_propagates_yolo_tracking_failure(
    mocker,
) -> None:
    tracking_error = RuntimeError(
        "YOLO tracking failed"
    )

    yolo_model = (
        _create_yolo_model_without_trackers(
            mocker=mocker,
        )
    )
    yolo_model.track.side_effect = (
        tracking_error
    )

    clip_as_numpy = [
        np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    ]

    results_stream = track_video(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device="cpu",
    )

    # The failure occurs when the lazy iterator starts tracking.
    with pytest.raises(
        RuntimeError,
        match="YOLO tracking failed",
    ) as exception_info:
        list(results_stream)

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
        is clip_as_numpy[0]
    )
    assert call_arguments["stream"] is True
    assert (
        call_arguments["tracker"]
        == "bytetrack.yaml"
    )
    assert call_arguments["device"] == "cpu"
    assert call_arguments["persist"] is True
