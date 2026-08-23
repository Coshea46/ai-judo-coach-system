import ffmpeg
import pytest

from video.input_video_cleanse.strip_audio import strip_audio
from exceptions.video import InvalidVideoError


def test_strip_audio_returns_temp_path(mocker):
    mock_input = mocker.patch("video.input_video_cleanse.strip_audio.ffmpeg.input")
    mock_input.return_value.output.return_value.overwrite_output.return_value.run.return_value = None

    result_path = strip_audio("input.mp4")

    assert result_path.endswith(".mp4")
    assert result_path != "input.mp4"


def test_strip_audio_raises_on_ffmpeg_error(mocker):
    mocker.patch(
        "video.input_video_cleanse.strip_audio.ffmpeg.input",
        side_effect=ffmpeg.Error("input", "stdout", "stderr"),
    )

    with pytest.raises(InvalidVideoError):
        strip_audio("input.mp4")