from video.input_video_cleanse.cleanse_input import cleanse_input_video


def test_cleanse_input_video_calls_strip_then_normalize_in_order(mocker):
    mock_strip = mocker.patch(
        "video.input_video_cleanse.cleanse_input.strip_audio",
        return_value="no_audio.mp4",
    )
    mock_normalize = mocker.patch(
        "video.input_video_cleanse.cleanse_input.normalize_video_fps",
        return_value="final.mp4",
    )

    result = cleanse_input_video("input.mp4")

    mock_strip.assert_called_once_with(input_video_path="input.mp4")
    mock_normalize.assert_called_once_with(
        input_video_path="no_audio.mp4", target_fps=30
    )
    assert result == "final.mp4"