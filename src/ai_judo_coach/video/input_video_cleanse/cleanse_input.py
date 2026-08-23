import math
from pathlib import Path
from typing import Any

import ffmpeg

from ai_judo_coach.config import TARGET_FPS
from ai_judo_coach.exceptions import InvalidVideoError

from .normalize import normalize_video_fps
from .strip_audio import strip_audio


def cleanse_input_video(
    input_video_path: str,
    output_directory: str,
    target_fps: float = TARGET_FPS,
) -> tuple[str, float]:
    """
    Convert the input video into the format expected by the pipeline.

    The pipeline expects a video with no audio and a fixed frame rate.
    Generated files are stored in the supplied job output directory.

    Returns:
        A pair containing the cleansed video path and its duration
        in seconds.
    """

    cleanse_output_directory = (
        Path(output_directory)
        / "input_cleanse"
    )

    cleanse_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    no_audio_video_path = (
        cleanse_output_directory
        / "input_without_audio.mp4"
    )

    cleansed_video_path = (
        cleanse_output_directory
        / "cleansed_input.mp4"
    )

    strip_audio(
        input_video_path=input_video_path,
        output_video_path=str(no_audio_video_path),
    )

    try:
        normalize_video_fps(
            input_video_path=str(no_audio_video_path),
            target_fps=target_fps,
            output_video_path=str(cleansed_video_path),
        )
    finally:
        # The intermediate audio-free file is no longer needed after
        # frame-rate normalisation, including when normalisation fails.
        no_audio_video_path.unlink(missing_ok=True)

    source_video_duration = _get_video_duration_seconds(
        video_path=cleansed_video_path,
    )

    return (
        str(cleansed_video_path),
        source_video_duration,
    )


def _get_video_duration_seconds(
    video_path: Path,
) -> float:
    """Return the duration of a video in seconds using FFprobe."""

    if not video_path.is_file():
        raise InvalidVideoError(
            f"Video does not exist: {video_path}"
        )

    try:
        probe_result: dict[str, Any] = ffmpeg.probe(
            str(video_path)
        )
    except ffmpeg.Error as error:
        error_message = (
            error.stderr.decode(errors="replace")
            if error.stderr
            else str(error)
        )

        raise InvalidVideoError(
            "Unable to determine the cleansed video duration: "
            f"{error_message}"
        ) from error

    format_metadata = probe_result.get("format")

    if not isinstance(format_metadata, dict):
        raise InvalidVideoError(
            "FFprobe output does not contain video format metadata"
        )

    raw_duration = format_metadata.get("duration")

    try:
        duration_seconds = float(raw_duration)
    except (TypeError, ValueError) as error:
        raise InvalidVideoError(
            "FFprobe returned an invalid video duration: "
            f"{raw_duration!r}"
        ) from error

    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0.0
    ):
        raise InvalidVideoError(
            "Cleansed video duration must be a finite number "
            f"greater than zero, got {duration_seconds!r}"
        )

    return duration_seconds
