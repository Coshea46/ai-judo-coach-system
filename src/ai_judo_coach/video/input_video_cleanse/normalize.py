from pathlib import Path

import ffmpeg

from ai_judo_coach.exceptions import InvalidVideoError


def normalize_video_fps(
    input_video_path: str,
    target_fps: float,
    output_video_path: str,
) -> str:
    """
    Re-encode the input video to a fixed FPS so downstream pipeline
    stages can assume consistent frame timing.

    The normalised video is written to the supplied output path.
    """

    if target_fps <= 0.0:
        raise ValueError(
            "target_fps must be greater than zero"
        )

    input_path = Path(input_video_path)
    output_path = Path(output_video_path)

    if not input_path.is_file():
        raise InvalidVideoError(
            f"Input video does not exist: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        (
            ffmpeg.input(str(input_path))
            .filter(
                "fps",
                fps=target_fps,
            )
            .output(str(output_path))
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as error:
        output_path.unlink(missing_ok=True)

        error_message = (
            error.stderr.decode(errors="replace")
            if error.stderr
            else str(error)
        )

        raise InvalidVideoError(
            "Unable to convert input video to target FPS: "
            f"{error_message}"
        ) from error

    if not output_path.is_file():
        raise InvalidVideoError(
            "FFmpeg completed without creating the normalised video: "
            f"{output_path}"
        )

    return str(output_path)
