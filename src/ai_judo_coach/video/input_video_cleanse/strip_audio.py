from pathlib import Path

import ffmpeg

from ai_judo_coach.exceptions import InvalidVideoError


def strip_audio(
    input_video_path: str,
    output_video_path: str,
    video_output_options: dict[str, object] | None = None,
) -> str:
    """
    Remove the audio track from the input video.

    The audio-free video is written to the supplied output path.
    """

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

    output_options = dict(
        video_output_options or {}
    )
    output_options["an"] = None

    try:
        (
            ffmpeg.input(str(input_path))
            .output(
                str(output_path),
                **output_options,
            )
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
            "Unable to remove audio from the input video: "
            f"{error_message}"
        ) from error

    if not output_path.is_file():
        raise InvalidVideoError(
            "FFmpeg completed without creating the audio-free video: "
            f"{output_path}"
        )

    return str(output_path)
