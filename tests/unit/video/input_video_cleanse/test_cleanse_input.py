from pathlib import Path
from unittest.mock import call

import pytest

from ai_judo_coach.exceptions import InvalidVideoError
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
        side_effect=[
            21.0,
            21.0,
        ],
    )

    output_directory = tmp_path / "job"

    result = cleanse_input_video(
        input_video_path="input.mp4",
        output_directory=str(
            output_directory
        ),
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
        output_video_path=str(
            no_audio_path
        ),
    )

    mock_normalize.assert_called_once_with(
        input_video_path=str(
            no_audio_path
        ),
        target_fps=30.0,
        output_video_path=str(
            cleansed_path
        ),
    )

    assert mock_duration.call_args_list == [
        call(
            video_path=Path(
                "input.mp4"
            ),
        ),
        call(
            video_path=cleansed_path,
        ),
    ]

    assert result == (
        str(cleansed_path),
        21.0,
    )


def test_cleanse_input_video_accepts_exactly_thirty_minutes(
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
        side_effect=[
            1800.0,
            1800.0,
        ],
    )

    output_directory = tmp_path / "job"

    result = cleanse_input_video(
        input_video_path="input.mp4",
        output_directory=str(
            output_directory
        ),
    )

    cleansed_path = (
        output_directory
        / "input_cleanse"
        / "cleansed_input.mp4"
    )

    assert result == (
        str(cleansed_path),
        1800.0,
    )

    mock_strip.assert_called_once()
    mock_normalize.assert_called_once()

    assert mock_duration.call_args_list == [
        call(
            video_path=Path(
                "input.mp4"
            ),
        ),
        call(
            video_path=cleansed_path,
        ),
    ]


def test_cleanse_input_video_rejects_video_over_thirty_minutes_before_ffmpeg(
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
        return_value=1800.001,
    )

    output_directory = tmp_path / "job"

    with pytest.raises(
        InvalidVideoError,
        match="must not exceed 30 minutes",
    ):
        cleanse_input_video(
            input_video_path="input.mp4",
            output_directory=str(
                output_directory
            ),
        )

    mock_duration.assert_called_once_with(
        video_path=Path(
            "input.mp4"
        ),
    )

    mock_strip.assert_not_called()
    mock_normalize.assert_not_called()

    assert not output_directory.exists()
