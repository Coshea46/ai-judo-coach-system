from fractions import Fraction
from pathlib import Path

import ffmpeg
import pytest

from ai_judo_coach.config import (
    MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC,
    MAX_GENERATED_ATTEMPT_CLIPS,
    OUTPUT_CLIP_NAMING_PATTERN,
    TARGET_FPS,
)
from ai_judo_coach.pipeline.orchestrator import (
    run_pipeline,
)
from ai_judo_coach.schemas.internal import (
    GeneratedAttemptClip,
)


pytestmark = pytest.mark.integration


def _get_video_stream(
    video_path: Path,
) -> dict:
    """Probe a video and return its first video stream."""

    probe_result = ffmpeg.probe(
        str(video_path)
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


def _get_probed_duration_seconds(
    video_path: Path,
) -> float:
    """Probe and return a video's duration in seconds."""

    probe_result = ffmpeg.probe(
        str(video_path)
    )

    format_duration = (
        probe_result
        .get("format", {})
        .get("duration")
    )

    if format_duration is not None:
        return float(format_duration)

    video_stream = _get_video_stream(
        video_path=video_path,
    )
    stream_duration = video_stream.get(
        "duration"
    )

    assert stream_duration is not None, (
        f"No duration found for {video_path}"
    )

    return float(stream_duration)


def _get_stream_fps(
    video_stream: dict,
) -> float:
    """Return the average frame rate from an FFprobe stream."""

    frame_rate = video_stream.get(
        "avg_frame_rate"
    )

    assert frame_rate not in (
        None,
        "0/0",
    )

    return float(
        Fraction(frame_rate)
    )


def _assert_generated_clips_are_valid(
    generated_clips: list[GeneratedAttemptClip],
    temporary_output_directory: Path,
) -> None:
    """Validate generated clip descriptors and real MP4 outputs."""

    generated_clips_directory = (
        temporary_output_directory
        / "generated_clips"
    )

    assert generated_clips_directory.is_dir()

    assert 1 <= len(generated_clips) <= (
        MAX_GENERATED_ATTEMPT_CLIPS
    )

    expected_clip_ids = [
        str(clip_index)
        for clip_index in range(
            len(generated_clips)
        )
    ]

    assert [
        generated_clip.clip_id
        for generated_clip in generated_clips
    ] == expected_clip_ids

    assert generated_clips == sorted(
        generated_clips,
        key=lambda generated_clip: (
            generated_clip.start_time_seconds,
            generated_clip.end_time_seconds,
        ),
    )

    returned_file_paths: list[Path] = []

    for generated_clip in generated_clips:
        assert isinstance(
            generated_clip,
            GeneratedAttemptClip,
        )

        assert (
            generated_clip.start_time_seconds
            >= 0.0
        )
        assert (
            generated_clip.end_time_seconds
            > generated_clip.start_time_seconds
        )

        selected_duration = (
            generated_clip.end_time_seconds
            - generated_clip.start_time_seconds
        )

        assert selected_duration <= (
            MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC
            + 1e-6
        )

        generated_file_path = Path(
            generated_clip.file_path
        ).resolve()

        returned_file_paths.append(
            generated_file_path
        )

        assert generated_file_path.is_file()
        assert generated_file_path.stat().st_size > 0

        assert (
            generated_file_path.parent
            == generated_clips_directory.resolve()
        )

        expected_filename = (
            f"{OUTPUT_CLIP_NAMING_PATTERN}"
            f"{int(generated_clip.clip_id):03d}.mp4"
        )

        assert (
            generated_file_path.name
            == expected_filename
        )

        video_stream = _get_video_stream(
            video_path=generated_file_path,
        )

        assert _get_stream_fps(
            video_stream=video_stream,
        ) == pytest.approx(
            TARGET_FPS,
            abs=0.01,
        )

        probed_duration = (
            _get_probed_duration_seconds(
                video_path=generated_file_path,
            )
        )

        assert probed_duration == pytest.approx(
            selected_duration,
            abs=0.15,
        )

        probe_result = ffmpeg.probe(
            str(generated_file_path)
        )

        audio_streams = [
            stream
            for stream in probe_result["streams"]
            if stream.get("codec_type") == "audio"
        ]

        assert audio_streams == []

    files_on_disk = sorted(
        generated_clips_directory.glob(
            "*.mp4"
        )
    )

    assert [
        file_path.resolve()
        for file_path in files_on_disk
    ] == sorted(returned_file_paths)


def test_run_pipeline_generates_clip_for_known_attempt(
    attempt_video_path: Path,
    tmp_path: Path,
) -> None:
    result = run_pipeline(
        input_video_path=str(
            attempt_video_path
        ),
        temporary_output_directory=str(
            tmp_path
        ),
    )

    assert result

    _assert_generated_clips_are_valid(
        generated_clips=result,
        temporary_output_directory=tmp_path,
    )

    cleansed_video_path = (
        tmp_path
        / "input_cleanse"
        / "cleansed_input.mp4"
    )

    assert cleansed_video_path.is_file()
    assert cleansed_video_path.stat().st_size > 0


def test_run_pipeline_returns_empty_list_for_known_no_throw_clip(
    no_throw_video_path: Path,
    tmp_path: Path,
) -> None:
    result = run_pipeline(
        input_video_path=str(
            no_throw_video_path
        ),
        temporary_output_directory=str(
            tmp_path
        ),
    )

    assert result == []

    cleansed_video_path = (
        tmp_path
        / "input_cleanse"
        / "cleansed_input.mp4"
    )

    assert cleansed_video_path.is_file()
    assert cleansed_video_path.stat().st_size > 0

    generated_clips_directory = (
        tmp_path
        / "generated_clips"
    )

    assert not generated_clips_directory.exists()
