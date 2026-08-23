from pathlib import Path

import ffmpeg

from ai_judo_coach.schemas.internal import (
    GeneratedAttemptClip,
    SelectedInterval,
)


def extract_final_clips(
    selected_intervals: list[SelectedInterval],
    temporary_output_dir_path: str,
    clip_naming_pattern: str,
    source_video_path: str,
    desired_fps: float,
) -> list[GeneratedAttemptClip]:
    """
    Extract the selected intervals from the source video.

    Generated MP4 files are stored in the supplied temporary output
    directory and returned as GeneratedAttemptClip instances.
    """

    source_path = Path(source_video_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source video does not exist: {source_path}"
        )

    if desired_fps <= 0.0:
        raise ValueError(
            "desired_fps must be greater than zero"
        )

    output_directory = Path(
        temporary_output_dir_path
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_clips: list[GeneratedAttemptClip] = []

    for selected_interval in selected_intervals:
        generated_clip = _extract_single_clip(
            selected_interval=selected_interval,
            source_video_path=str(source_path),
            desired_fps=desired_fps,
            clip_naming_pattern=clip_naming_pattern,
            output_dir_path=str(output_directory),
        )

        generated_clips.append(generated_clip)

    return generated_clips


def _extract_single_clip(
    selected_interval: SelectedInterval,
    source_video_path: str,
    desired_fps: float,
    clip_naming_pattern: str,
    output_dir_path: str,
) -> GeneratedAttemptClip:
    """Extract one selected interval as an MP4 file."""

    output_path = _construct_output_path(
        clip_naming_pattern=clip_naming_pattern,
        clip_id=selected_interval.clip_id,
        output_dir_path=output_dir_path,
    )

    interval_duration = (
        selected_interval.end_time_seconds
        - selected_interval.start_time_seconds
    )

    if interval_duration <= 0.0:
        raise ValueError(
            "Selected interval duration must be greater than zero: "
            f"start={selected_interval.start_time_seconds}, "
            f"end={selected_interval.end_time_seconds}"
        )

    try:
        (
            ffmpeg
            .input(
                source_video_path,
                ss=selected_interval.start_time_seconds,
            )
            .output(
                output_path,
                t=interval_duration,
                vcodec="libx264",
                preset="fast",
                an=None,
                r=desired_fps,
            )
            .overwrite_output()
            .run(
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as error:
        Path(output_path).unlink(
            missing_ok=True,
        )

        error_message = (
            error.stderr.decode(errors="replace")
            if error.stderr
            else str(error)
        )

        raise RuntimeError(
            "Unable to extract attempt clip "
            f"{selected_interval.clip_id}: "
            f"{error_message}"
        ) from error

    if not Path(output_path).is_file():
        raise RuntimeError(
            "FFmpeg completed without creating the expected clip: "
            f"{output_path}"
        )

    return GeneratedAttemptClip(
        clip_id=selected_interval.clip_id,
        start_time_seconds=(
            selected_interval.start_time_seconds
        ),
        end_time_seconds=(
            selected_interval.end_time_seconds
        ),
        file_path=output_path,
    )


def _construct_output_path(
    clip_naming_pattern: str,
    clip_id: str,
    output_dir_path: str,
) -> str:
    """
    Constructs output path within
    temporary directory for
    a given identified attempt clip.
    """

    try:
        numeric_clip_id = int(clip_id)
    except ValueError as error:
        raise ValueError(
            "clip_id must contain a numeric value"
        ) from error

    clip_basename = (
        f"{clip_naming_pattern}"
        f"{numeric_clip_id:03d}.mp4"
    )

    return str(
        Path(output_dir_path)
        / clip_basename
    )
