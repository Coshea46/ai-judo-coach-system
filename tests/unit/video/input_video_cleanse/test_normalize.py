import ffmpeg
import pytest

from video.input_video_cleanse.normalize import normalize_video_fps
from exceptions.video import InvalidVideoError


def test_normalize_video_fps_returns_temp_path(mocker):
    mock_run = mocker.patch("video.input_video_cleanse.normalize.ffmpeg.input")
    mock_run.return_value.filter.return_value.output.return_value.overwrite_output.return_value.run.return_value = None

    result_path = normalize_video_fps("input.mp4", target_fps=30)

    assert result_path.endswith(".mp4")
    assert result_path != "input.mp4"


def test_normalize_video_fps_raises_on_ffmpeg_error(mocker):
    mocker.patch(
        "video.input_video_cleanse.normalize.ffmpeg.input",
        side_effect=ffmpeg.Error("input", "stdout", "stderr"),
    )

    with pytest.raises(InvalidVideoError):
        normalize_video_fps("input.mp4", target_fps=30)