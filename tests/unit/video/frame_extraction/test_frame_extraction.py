from unittest.mock import call

import numpy as np
import pytest

from ai_judo_coach.exceptions import (
    InvalidFrameIndicesError,
)
from ai_judo_coach.schemas.internal import (
    InitialClipWindow,
)
from ai_judo_coach.video.frame_extraction.frame_extraction import (
    _check_frame_indices,
    _parse_desired_device,
    _read_video,
    extract_frames_from_initial_window,
)


FRAME_EXTRACTION_MODULE_PATH = (
    "ai_judo_coach.video.frame_extraction."
    "frame_extraction"
)


class _FakeBatch:
    """Minimal substitute for a Decord batch."""

    def __init__(
        self,
        frames: np.ndarray,
    ) -> None:
        self.frames = frames

    def asnumpy(self) -> np.ndarray:
        return self.frames


class _TestDecordError(Exception):
    """Controllable replacement for Decord's exception type."""


def test_extract_frames_from_initial_window_extracts_expected_frames_in_bgr(
    mocker,
) -> None:
    rgb_frames = np.array(
        [
            [
                [
                    [10, 20, 30],
                    [40, 50, 60],
                ],
            ],
            [
                [
                    [70, 80, 90],
                    [100, 110, 120],
                ],
            ],
        ],
        dtype=np.uint8,
    )

    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = 10
    video_reader.get_batch.return_value = (
        _FakeBatch(rgb_frames)
    )

    read_video_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    window = InitialClipWindow(
        start_time=1.0,
        end_time=2.0,
        window_id=4,
    )

    result = extract_frames_from_initial_window(
        source_video_path="/videos/source.mp4",
        window=window,
        video_fps=2.0,
        device="gpu:0",
    )

    read_video_mock.assert_called_once_with(
        source_video_path="/videos/source.mp4",
        desired_device="gpu:0",
    )

    video_reader.get_batch.assert_called_once_with(
        indices=[
            2,
            3,
        ],
    )

    assert isinstance(result, list)
    assert len(result) == 2

    np.testing.assert_array_equal(
        result[0],
        rgb_frames[0, :, :, ::-1],
    )
    np.testing.assert_array_equal(
        result[1],
        rgb_frames[1, :, :, ::-1],
    )

    assert result[0].dtype == np.uint8
    assert result[1].dtype == np.uint8
    assert result[0].flags.c_contiguous
    assert result[1].flags.c_contiguous


def test_extract_frames_from_initial_window_returns_copies_of_rgb_data(
    mocker,
) -> None:
    rgb_frames = np.array(
        [
            [
                [
                    [10, 20, 30],
                ],
            ],
        ],
        dtype=np.uint8,
    )

    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = 1
    video_reader.get_batch.return_value = (
        _FakeBatch(rgb_frames)
    )

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    result = extract_frames_from_initial_window(
        source_video_path="/videos/source.mp4",
        window=InitialClipWindow(
            start_time=0.0,
            end_time=1.0,
            window_id=0,
        ),
        video_fps=1.0,
        device="cpu",
    )

    result[0][0, 0, 0] = 255

    np.testing.assert_array_equal(
        rgb_frames,
        np.array(
            [
                [
                    [
                        [10, 20, 30],
                    ],
                ],
            ],
            dtype=np.uint8,
        ),
    )


def test_extract_frames_from_seven_second_window_returns_210_frames(
    mocker,
) -> None:
    frame_count = 210

    rgb_frames = np.zeros(
        (
            frame_count,
            2,
            2,
            3,
        ),
        dtype=np.uint8,
    )

    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = frame_count
    video_reader.get_batch.return_value = (
        _FakeBatch(rgb_frames)
    )

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    result = extract_frames_from_initial_window(
        source_video_path="/videos/source.mp4",
        window=InitialClipWindow(
            start_time=0.0,
            end_time=7.0,
            window_id=0,
        ),
        video_fps=30.0,
        device="cpu",
    )

    assert len(result) == 210

    video_reader.get_batch.assert_called_once_with(
        indices=list(range(210)),
    )


def test_extract_frames_accepts_window_ending_at_final_video_frame(
    mocker,
) -> None:
    rgb_frames = np.zeros(
        (
            5,
            2,
            2,
            3,
        ),
        dtype=np.uint8,
    )

    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = 5
    video_reader.get_batch.return_value = (
        _FakeBatch(rgb_frames)
    )

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    result = extract_frames_from_initial_window(
        source_video_path="/videos/source.mp4",
        window=InitialClipWindow(
            start_time=0.0,
            end_time=5.0,
            window_id=0,
        ),
        video_fps=1.0,
        device="cpu",
    )

    assert len(result) == 5

    video_reader.get_batch.assert_called_once_with(
        indices=[
            0,
            1,
            2,
            3,
            4,
        ],
    )


def test_extract_frames_rejects_window_extending_beyond_video(
    mocker,
) -> None:
    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = 5

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    with pytest.raises(
        InvalidFrameIndicesError,
        match="Desired end frame index out of bounds",
    ):
        extract_frames_from_initial_window(
            source_video_path="/videos/source.mp4",
            window=InitialClipWindow(
                start_time=0.0,
                end_time=6.0,
                window_id=0,
            ),
            video_fps=1.0,
            device="cpu",
        )

    video_reader.get_batch.assert_not_called()


def test_extract_frames_rejects_negative_window_start(
    mocker,
) -> None:
    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = 10

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    with pytest.raises(
        InvalidFrameIndicesError,
        match="Desired start frame index out of bounds",
    ):
        extract_frames_from_initial_window(
            source_video_path="/videos/source.mp4",
            window=InitialClipWindow(
                start_time=-1.0,
                end_time=2.0,
                window_id=0,
            ),
            video_fps=1.0,
            device="cpu",
        )

    video_reader.get_batch.assert_not_called()


def test_extract_frames_rejects_end_before_start(
    mocker,
) -> None:
    video_reader = mocker.MagicMock()
    video_reader.__len__.return_value = 10

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "_read_video",
        return_value=video_reader,
    )

    with pytest.raises(
        InvalidFrameIndicesError,
        match=(
            "Desired end frame index precedes "
            "start frame index"
        ),
    ):
        extract_frames_from_initial_window(
            source_video_path="/videos/source.mp4",
            window=InitialClipWindow(
                start_time=4.0,
                end_time=3.0,
                window_id=0,
            ),
            video_fps=1.0,
            device="cpu",
        )

    video_reader.get_batch.assert_not_called()


def test_read_video_constructs_reader_with_cpu_context(
    mocker,
) -> None:
    cpu_context = object()
    expected_reader = mocker.Mock()

    cpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
        return_value=cpu_context,
    )
    gpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
    )
    video_reader_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "VideoReader",
        return_value=expected_reader,
    )

    result = _read_video(
        source_video_path="/videos/source.mp4",
        desired_device=" CPU ",
    )

    assert result is expected_reader

    cpu_mock.assert_called_once_with(0)
    gpu_mock.assert_not_called()
    video_reader_mock.assert_called_once_with(
        uri="/videos/source.mp4",
        ctx=cpu_context,
    )


@pytest.mark.parametrize(
    (
        "desired_device",
        "expected_device_id",
    ),
    [
        (
            "gpu",
            0,
        ),
        (
            "GPU:2",
            2,
        ),
        (
            "cuda",
            0,
        ),
        (
            " CUDA:3 ",
            3,
        ),
    ],
)
def test_read_video_constructs_reader_with_gpu_context(
    mocker,
    desired_device: str,
    expected_device_id: int,
) -> None:
    gpu_context = object()
    expected_reader = mocker.Mock()

    gpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
        return_value=gpu_context,
    )
    cpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
    )
    video_reader_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "VideoReader",
        return_value=expected_reader,
    )

    result = _read_video(
        source_video_path="/videos/source.mp4",
        desired_device=desired_device,
    )

    assert result is expected_reader

    gpu_mock.assert_called_once_with(
        expected_device_id
    )
    cpu_mock.assert_not_called()
    video_reader_mock.assert_called_once_with(
        uri="/videos/source.mp4",
        ctx=gpu_context,
    )


@pytest.mark.parametrize(
    "desired_device",
    [
        "gpu",
        "gpu:1",
        "cuda",
        "cuda:2",
    ],
)
def test_read_video_falls_back_to_cpu_when_gpu_reader_fails(
    mocker,
    desired_device: str,
) -> None:
    gpu_context = object()
    cpu_context = object()
    expected_reader = mocker.Mock()

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "DECORDError",
        new=_TestDecordError,
    )

    gpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
        return_value=gpu_context,
    )
    cpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
        return_value=cpu_context,
    )

    video_reader_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "VideoReader",
        side_effect=[
            _TestDecordError(
                "GPU decoding unavailable"
            ),
            expected_reader,
        ],
    )

    result = _read_video(
        source_video_path="/videos/source.mp4",
        desired_device=desired_device,
    )

    assert result is expected_reader
    gpu_mock.assert_called_once()
    cpu_mock.assert_called_once_with(0)

    assert video_reader_mock.call_args_list == [
        call(
            uri="/videos/source.mp4",
            ctx=gpu_context,
        ),
        call(
            uri="/videos/source.mp4",
            ctx=cpu_context,
        ),
    ]


def test_read_video_does_not_retry_cpu_when_cpu_reader_fails(
    mocker,
) -> None:
    cpu_context = object()
    decord_error = _TestDecordError(
        "Unable to read video"
    )

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "DECORDError",
        new=_TestDecordError,
    )
    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
        return_value=cpu_context,
    )
    video_reader_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "VideoReader",
        side_effect=decord_error,
    )

    with pytest.raises(
        _TestDecordError,
        match="Unable to read video",
    ) as exception_info:
        _read_video(
            source_video_path="/videos/source.mp4",
            desired_device="cpu",
        )

    assert exception_info.value is decord_error

    video_reader_mock.assert_called_once_with(
        uri="/videos/source.mp4",
        ctx=cpu_context,
    )


def test_read_video_propagates_cpu_fallback_failure(
    mocker,
) -> None:
    gpu_context = object()
    cpu_context = object()

    gpu_error = _TestDecordError(
        "GPU decoding unavailable"
    )
    cpu_error = _TestDecordError(
        "CPU decoding also failed"
    )

    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "DECORDError",
        new=_TestDecordError,
    )
    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
        return_value=gpu_context,
    )
    mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
        return_value=cpu_context,
    )
    video_reader_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}."
        "VideoReader",
        side_effect=[
            gpu_error,
            cpu_error,
        ],
    )

    with pytest.raises(
        _TestDecordError,
        match="CPU decoding also failed",
    ) as exception_info:
        _read_video(
            source_video_path="/videos/source.mp4",
            desired_device="cuda:0",
        )

    assert exception_info.value is cpu_error

    assert video_reader_mock.call_args_list == [
        call(
            uri="/videos/source.mp4",
            ctx=gpu_context,
        ),
        call(
            uri="/videos/source.mp4",
            ctx=cpu_context,
        ),
    ]


@pytest.mark.parametrize(
    (
        "device_string",
        "expected_device_id",
    ),
    [
        (
            "gpu",
            0,
        ),
        (
            "GPU",
            0,
        ),
        (
            " gpu:1 ",
            1,
        ),
        (
            "cuda",
            0,
        ),
        (
            "CUDA:4",
            4,
        ),
    ],
)
def test_parse_desired_device_returns_requested_gpu_context(
    mocker,
    device_string: str,
    expected_device_id: int,
) -> None:
    expected_context = object()

    gpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
        return_value=expected_context,
    )
    cpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
    )

    result = _parse_desired_device(
        device_str=device_string,
    )

    assert result is expected_context
    gpu_mock.assert_called_once_with(
        expected_device_id
    )
    cpu_mock.assert_not_called()


@pytest.mark.parametrize(
    "device_string",
    [
        "cpu",
        " CPU ",
        "mps",
        "",
        "unknown",
    ],
)
def test_parse_desired_device_defaults_non_gpu_values_to_cpu(
    mocker,
    device_string: str,
) -> None:
    expected_context = object()

    cpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
        return_value=expected_context,
    )
    gpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
    )

    result = _parse_desired_device(
        device_str=device_string,
    )

    assert result is expected_context
    cpu_mock.assert_called_once_with(0)
    gpu_mock.assert_not_called()


def test_parse_desired_device_rejects_non_numeric_gpu_id(
    mocker,
) -> None:
    gpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.gpu",
    )
    cpu_mock = mocker.patch(
        f"{FRAME_EXTRACTION_MODULE_PATH}.cpu",
    )

    with pytest.raises(ValueError):
        _parse_desired_device(
            device_str="cuda:not-a-number",
        )

    gpu_mock.assert_not_called()
    cpu_mock.assert_not_called()


def test_check_frame_indices_accepts_complete_range() -> None:
    _check_frame_indices(
        desired_start_frame_idx=3,
        desired_end_frame_idx=5,
        found_frame_indices=[
            3,
            4,
            5,
        ],
    )


@pytest.mark.parametrize(
    (
        "desired_start_frame_idx",
        "desired_end_frame_idx",
        "found_frame_indices",
        "expected_message",
    ),
    [
        (
            -1,
            5,
            [-1, 0, 1, 2, 3, 4, 5],
            "Desired start frame index out of bounds",
        ),
        (
            0,
            -1,
            [],
            "Desired end frame index out of bounds",
        ),
        (
            5,
            4,
            [],
            (
                "Desired end frame index precedes "
                "start frame index"
            ),
        ),
        (
            3,
            5,
            [4, 5],
            "Desired start frame index out of bounds",
        ),
        (
            3,
            5,
            [3, 4],
            "Desired end frame index out of bounds",
        ),
    ],
)
def test_check_frame_indices_rejects_invalid_indices(
    desired_start_frame_idx: int,
    desired_end_frame_idx: int,
    found_frame_indices: list[int],
    expected_message: str,
) -> None:
    with pytest.raises(
        InvalidFrameIndicesError,
        match=expected_message,
    ):
        _check_frame_indices(
            desired_start_frame_idx=(
                desired_start_frame_idx
            ),
            desired_end_frame_idx=(
                desired_end_frame_idx
            ),
            found_frame_indices=(
                found_frame_indices
            ),
        )
