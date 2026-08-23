from ai_judo_coach.video.input_video_cleanse import (
    cleanse_input_video,
)


CLEANSE_MODULE_PATH = (
    "ai_judo_coach.video.input_video_cleanse.cleanse_input"
)


def test_cleanse_input_video_calls_strip_then_normalize_in_order(
    mocker,
    tmp_path,
):
    mock_strip = mocker.patch(
        f"{CLEANSE_MODULE_PATH}.strip_audio",
    )

    mock_normalize = mocker.patch(
        f"{CLEANSE_MODULE_PATH}.normalize_video_fps",
    )

    mock_duration = mocker.patch(
        f"{CLEANSE_MODULE_PATH}._get_video_duration_seconds",
        return_value=21.0,
    )

    output_directory = tmp_path / "job"

    result = cleanse_input_video(
        input_video_path="input.mp4",
        output_directory=str(output_directory),
    )

    no_audio_path = (
        output_directory
        / "input_cleanse"
        / "input_without_audio.mp4"
    )

    cleansed_path = (
        output_directory
        / "input_cleanse"
        / "cleansed_input.mp4"
    )

    mock_strip.assert_called_once_with(
        input_video_path="input.mp4",
        output_video_path=str(no_audio_path),
    )

    mock_normalize.assert_called_once_with(
        input_video_path=str(no_audio_path),
        target_fps=30.0,
        output_video_path=str(cleansed_path),
    )

    mock_duration.assert_called_once_with(
        video_path=cleansed_path,
    )

    assert result == (
        str(cleansed_path),
        21.0,
    )
