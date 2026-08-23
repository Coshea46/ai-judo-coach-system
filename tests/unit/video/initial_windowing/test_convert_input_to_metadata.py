import ffmpeg
import pytest

from ai_judo_coach.exceptions import InvalidVideoError
from ai_judo_coach.video.initial_windowing import (
    compute_initial_clip_windows,
)


FFPROBE_PATCH_PATH = (
    "ai_judo_coach.video.initial_windowing."
    "convert_input_to_metadata.ffmpeg.probe"
)


def test_basic_windowing(mocker):
    mocker.patch(
        FFPROBE_PATCH_PATH,
        return_value={"format": {"duration": "21.0"}},
    )

    windows = list(
        compute_initial_clip_windows(
            "fake.mp4",
            7.0,
            7.0,
        )
    )

    assert len(windows) == 3
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 7.0
    assert windows[2].end_time == 21.0


def test_video_shorter_than_window_raises(mocker):
    mocker.patch(
        FFPROBE_PATCH_PATH,
        return_value={"format": {"duration": "3.0"}},
    )

    with pytest.raises(InvalidVideoError):
        list(
            compute_initial_clip_windows(
                "fake.mp4",
                7.0,
                7.0,
            )
        )


def test_remainder_window_overlaps_correctly(mocker):
    mocker.patch(
        FFPROBE_PATCH_PATH,
        return_value={"format": {"duration": "25.0"}},
    )

    windows = list(
        compute_initial_clip_windows(
            "fake.mp4",
            7.0,
            7.0,
        )
    )

    assert windows[-1].end_time == 25.0
    assert windows[-1].start_time == 18.0  # 25 - 7


def test_invalid_video_raises(mocker):
    mocker.patch(
        FFPROBE_PATCH_PATH,
        side_effect=ffmpeg.Error(
            "probe",
            b"stdout",
            b"stderr",
        ),
    )

    with pytest.raises(InvalidVideoError):
        list(
            compute_initial_clip_windows(
                "fake.mp4",
                7.0,
                7.0,
            )
        )
