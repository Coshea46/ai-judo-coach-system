import tempfile

import ffmpeg

from exceptions import InvalidVideoError


def normalize_video_fps(input_video_path: str, target_fps: int) -> str:
    """
    Re-encodes the input video to a fixed fps, so downstream pipeline
    stages can assume consistent frame timing regardless of source video.
    """

    normalized_fps_video_path = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ).name

    try:
        (
            ffmpeg.input(input_video_path)
            .filter("fps", fps=target_fps)
            .output(normalized_fps_video_path)
            .overwrite_output()
            .run(quiet=True)
        )

    except ffmpeg.Error as e:
        raise InvalidVideoError(
            f"Unable to convert input video to target fps: {e}"
        ) from e

    return normalized_fps_video_path
