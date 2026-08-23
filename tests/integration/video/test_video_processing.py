from fractions import Fraction
from pathlib import Path

import decord
import ffmpeg
import numpy as np
import pytest
from decord import VideoReader, cpu

from ai_judo_coach.config import (
    CLIP_DURATION_SEC,
    CLIP_STRIDE_SEC,
    TARGET_FPS,
)
from ai_judo_coach.video import (
    cleanse_input_video,
    compute_initial_clip_windows,
    extract_frames_from_initial_window,
)


pytestmark = pytest.mark.integration


def _probe_video(
    video_path: Path,
) -> dict:
    """Probe a real video using FFprobe."""

    return ffmpeg.probe(
        str(video_path)
    )


def _get_video_stream(
    video_path: Path,
) -> dict:
    """Return the first video stream from a real video."""

    probe_result = _probe_video(
        video_path=video_path,
    )

    video_streams = [
        stream
        for stream in probe_result["streams"]
        if stream.get("codec_type") == "video"
    ]

    assert video_streams, (
        f"No video stream found in {video_path}"
    )

    return video_streams[0]


def _get_video_duration_seconds(
    video_path: Path,
) -> float:
    """Return the duration of a real video in seconds."""

    probe_result = _probe_video(
        video_path=video_path,
    )

    format_duration = (
        probe_result
        .get("format", {})
        .get("duration")
    )

    if format_duration is not None:
        return float(format_duration)

    video_streams = [
        stream
        for stream in probe_result["streams"]
        if stream.get("codec_type") == "video"
    ]

    assert video_streams, (
        f"No video stream found in {video_path}"
    )

    stream_duration = video_streams[0].get(
        "duration"
    )

    assert stream_duration is not None, (
        f"No duration found for {video_path}"
    )

    return float(stream_duration)


def _get_video_fps(
    video_path: Path,
) -> float:
    """Return the average frame rate of a real video."""

    video_stream = _get_video_stream(
        video_path=video_path,
    )

    frame_rate = video_stream.get(
        "avg_frame_rate"
    )

    assert frame_rate not in (
        None,
        "0/0",
    ), (
        f"No valid frame rate found for {video_path}"
    )

    return float(
        Fraction(frame_rate)
    )


def _get_audio_streams(
    video_path: Path,
) -> list[dict]:
    """Return all audio streams found in a real video."""

    probe_result = _probe_video(
        video_path=video_path,
    )

    return [
        stream
        for stream in probe_result["streams"]
        if stream.get("codec_type") == "audio"
    ]


def test_cleanse_input_video_creates_normalised_video_without_audio(
    short_full_match_video_path: Path,
    tmp_path: Path,
) -> None:
    source_duration = (
        _get_video_duration_seconds(
            video_path=short_full_match_video_path,
        )
    )

    cleansed_video_path_string, returned_duration = (
        cleanse_input_video(
            input_video_path=str(
                short_full_match_video_path
            ),
            output_directory=str(
                tmp_path
            ),
            target_fps=TARGET_FPS,
        )
    )

    cleansed_video_path = Path(
        cleansed_video_path_string
    )

    expected_output_path = (
        tmp_path
        / "input_cleanse"
        / "cleansed_input.mp4"
    )

    assert cleansed_video_path == expected_output_path
    assert cleansed_video_path.is_file()
    assert cleansed_video_path.stat().st_size > 0

    assert (
        short_full_match_video_path.is_file()
    )

    assert _get_video_fps(
        video_path=cleansed_video_path,
    ) == pytest.approx(
        TARGET_FPS,
        abs=0.01,
    )

    assert _get_audio_streams(
        video_path=cleansed_video_path,
    ) == []

    probed_cleansed_duration = (
        _get_video_duration_seconds(
            video_path=cleansed_video_path,
        )
    )

    assert returned_duration == pytest.approx(
        probed_cleansed_duration,
        abs=0.05,
    )

    assert probed_cleansed_duration == pytest.approx(
        source_duration,
        abs=0.20,
    )

    intermediate_path = (
        tmp_path
        / "input_cleanse"
        / "input_without_audio.mp4"
    )

    assert not intermediate_path.exists()


def test_compute_windows_and_extract_real_bgr_frames(
    short_full_match_video_path: Path,
    tmp_path: Path,
) -> None:
    cleansed_video_path_string, cleansed_duration = (
        cleanse_input_video(
            input_video_path=str(
                short_full_match_video_path
            ),
            output_directory=str(
                tmp_path
            ),
            target_fps=TARGET_FPS,
        )
    )

    cleansed_video_path = Path(
        cleansed_video_path_string
    )

    minimum_duration_for_two_windows = (
        float(CLIP_DURATION_SEC)
        + float(CLIP_STRIDE_SEC)
    )

    assert cleansed_duration >= (
        minimum_duration_for_two_windows
    ), (
        "short_full_match_video.mp4 must be at least "
        f"{minimum_duration_for_two_windows} seconds long"
    )

    windows = list(
        compute_initial_clip_windows(
            input_video_path=str(
                cleansed_video_path
            ),
            individual_window_duration=float(
                CLIP_DURATION_SEC
            ),
            stride=float(
                CLIP_STRIDE_SEC
            ),
        )
    )

    assert len(windows) >= 2

    first_window = windows[0]
    second_window = windows[1]

    assert first_window.window_id == 0
    assert first_window.start_time == pytest.approx(
        0.0
    )
    assert first_window.end_time == pytest.approx(
        float(CLIP_DURATION_SEC)
    )

    assert second_window.window_id == 1
    assert second_window.start_time == pytest.approx(
        float(CLIP_STRIDE_SEC)
    )
    assert second_window.end_time == pytest.approx(
        float(CLIP_STRIDE_SEC)
        + float(CLIP_DURATION_SEC)
    )

    assert [
        window.window_id
        for window in windows
    ] == list(
        range(len(windows))
    )

    assert all(
        window.start_time >= 0.0
        for window in windows
    )
    assert all(
        window.end_time > window.start_time
        for window in windows
    )
    assert all(
        window.end_time
        <= cleansed_duration + 0.05
        for window in windows
    )

    extracted_frames = (
        extract_frames_from_initial_window(
            source_video_path=str(
                cleansed_video_path
            ),
            window=first_window,
            video_fps=float(
                TARGET_FPS
            ),
            device="cpu",
        )
    )

    expected_frame_count = int(
        float(CLIP_DURATION_SEC)
        * float(TARGET_FPS)
    )

    assert len(extracted_frames) == (
        expected_frame_count
    )

    first_extracted_frame = extracted_frames[0]
    last_extracted_frame = extracted_frames[-1]

    assert isinstance(
        first_extracted_frame,
        np.ndarray,
    )
    assert isinstance(
        last_extracted_frame,
        np.ndarray,
    )

    assert first_extracted_frame.ndim == 3
    assert first_extracted_frame.shape[2] == 3
    assert first_extracted_frame.dtype == np.uint8
    assert first_extracted_frame.flags.c_contiguous

    assert (
        last_extracted_frame.shape
        == first_extracted_frame.shape
    )
    assert last_extracted_frame.dtype == np.uint8
    assert last_extracted_frame.flags.c_contiguous

    decord.bridge.set_bridge(
        "native"
    )

    video_reader = VideoReader(
        uri=str(cleansed_video_path),
        ctx=cpu(0),
    )

    first_frame_index = int(
        first_window.start_time
        * float(TARGET_FPS)
    )
    last_frame_index = (
        int(
            first_window.end_time
            * float(TARGET_FPS)
        )
        - 1
    )

    expected_first_bgr_frame = (
        video_reader[first_frame_index]
        .asnumpy()[:, :, ::-1]
        .copy()
    )
    expected_last_bgr_frame = (
        video_reader[last_frame_index]
        .asnumpy()[:, :, ::-1]
        .copy()
    )

    np.testing.assert_array_equal(
        first_extracted_frame,
        expected_first_bgr_frame,
    )
    np.testing.assert_array_equal(
        last_extracted_frame,
        expected_last_bgr_frame,
    )
