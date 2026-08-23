import ffmpeg
import pytest

from ai_judo_coach.exceptions.video import (
    InvalidVideoError,
)
from ai_judo_coach.video.input_video_cleanse.strip_audio import (
    strip_audio,
)


FFMPEG_INPUT_PATCH_PATH = (
    "ai_judo_coach.video.input_video_cleanse."
    "strip_audio.ffmpeg.input"
)


def test_strip_audio_writes_to_supplied_path(
    mocker,
    tmp_path,
):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "without_audio.mp4"

    input_path.touch()

    mock_input = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH,
    )

    mock_ffmpeg_run = (
        mock_input
        .return_value
        .output
        .return_value
        .overwrite_output
        .return_value
        .run
    )

    mock_ffmpeg_run.side_effect = (
        lambda **kwargs: output_path.touch()
    )

    result_path = strip_audio(
        input_video_path=str(input_path),
        output_video_path=str(output_path),
    )

    assert result_path == str(output_path)
    assert output_path.is_file()


def test_strip_audio_raises_on_ffmpeg_error(
    mocker,
    tmp_path,
):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "without_audio.mp4"

    input_path.touch()

    mocker.patch(
        FFMPEG_INPUT_PATCH_PATH,
        side_effect=ffmpeg.Error(
            "input",
            b"stdout",
            b"stderr",
        ),
    )

    with pytest.raises(InvalidVideoError):
        strip_audio(
            input_video_path=str(input_path),
            output_video_path=str(output_path),
        )

    assert not output_path.exists()
