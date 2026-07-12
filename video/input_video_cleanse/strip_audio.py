import tempfile

import ffmpeg

from exceptions import InvalidVideoError


def strip_audio(input_video_path: str) -> str:
    """
    Removes the audio track from the video.
    Returns path to audio-free output file stored in system temp folder.
    """

    stripped_audio_video_path = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ).name

    try:
        (
            ffmpeg.input(input_video_path)
            .output(stripped_audio_video_path, an=None)
            .overwrite_output()
            .run(quiet=True)
        )

    except ffmpeg.Error as e:
        raise InvalidVideoError(f"failed to strip input video audio: {e}") from e

    return stripped_audio_video_path
