from fractions import Fraction
from pathlib import Path

import ffmpeg
import pytest

from ai_judo_coach.attempt_clip_generation import (
    extract_final_clips,
)
from ai_judo_coach.config import (
    OUTPUT_CLIP_NAMING_PATTERN,
    TARGET_FPS,
)
from ai_judo_coach.schemas.internal import (
    GeneratedAttemptClip,
    SelectedInterval,
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
    """Return the first video stream in a probed file."""

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
    """Return the probed duration of a video."""

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
    """Return the average frame rate of a video."""

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


def _assert_no_audio_stream(
    video_path: Path,
) -> None:
    """Assert that a video contains no audio stream."""

    probe_result = _probe_video(
        video_path=video_path,
    )

    audio_streams = [
        stream
        for stream in probe_result["streams"]
        if stream.get("codec_type") == "audio"
    ]

    assert audio_streams == []


def test_extract_final_clips_creates_real_mp4_files(
    short_full_match_video_path: Path,
    tmp_path: Path,
) -> None:
    source_duration = (
        _get_video_duration_seconds(
            video_path=short_full_match_video_path,
        )
    )

    assert source_duration >= 4.0, (
        "short_full_match_video.mp4 must be at least "
        "four seconds long"
    )

    output_directory = (
        tmp_path
        / "generated_clips"
    )

    selected_intervals = [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=0.25,
            end_time_seconds=1.25,
        ),
        SelectedInterval(
            clip_id="7",
            start_time_seconds=2.0,
            end_time_seconds=3.25,
        ),
    ]

    result = extract_final_clips(
        selected_intervals=selected_intervals,
        temporary_output_dir_path=str(
            output_directory
        ),
        clip_naming_pattern=(
            OUTPUT_CLIP_NAMING_PATTERN
        ),
        source_video_path=str(
            short_full_match_video_path
        ),
        desired_fps=TARGET_FPS,
    )

    expected_output_paths = [
        output_directory
        / (
            f"{OUTPUT_CLIP_NAMING_PATTERN}"
            "000.mp4"
        ),
        output_directory
        / (
            f"{OUTPUT_CLIP_NAMING_PATTERN}"
            "007.mp4"
        ),
    ]

    assert result == [
        GeneratedAttemptClip(
            clip_id="0",
            start_time_seconds=0.25,
            end_time_seconds=1.25,
            file_path=str(
                expected_output_paths[0]
            ),
        ),
        GeneratedAttemptClip(
            clip_id="7",
            start_time_seconds=2.0,
            end_time_seconds=3.25,
            file_path=str(
                expected_output_paths[1]
            ),
        ),
    ]

    assert output_directory.is_dir()

    for (
        generated_clip,
        expected_output_path,
    ) in zip(
        result,
        expected_output_paths,
        strict=True,
    ):
        assert expected_output_path.is_file()
        assert expected_output_path.stat().st_size > 0

        assert (
            Path(generated_clip.file_path)
            == expected_output_path
        )

        video_stream = _get_video_stream(
            video_path=expected_output_path,
        )

        assert (
            video_stream.get("codec_name")
            == "h264"
        )

        assert _get_video_fps(
            video_path=expected_output_path,
        ) == pytest.approx(
            TARGET_FPS,
            abs=0.01,
        )

        expected_duration = (
            generated_clip.end_time_seconds
            - generated_clip.start_time_seconds
        )

        actual_duration = (
            _get_video_duration_seconds(
                video_path=expected_output_path,
            )
        )

        assert actual_duration == pytest.approx(
            expected_duration,
            abs=0.15,
        )

        _assert_no_audio_stream(
            video_path=expected_output_path,
        )

    assert sorted(
        output_directory.glob("*.mp4")
    ) == expected_output_paths
