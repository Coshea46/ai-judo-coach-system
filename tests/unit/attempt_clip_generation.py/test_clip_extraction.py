from pathlib import Path
from unittest.mock import call

import ffmpeg
import pytest

from ai_judo_coach.attempt_clip_generation.clip_extraction import (
    extract_final_clips,
)
from ai_judo_coach.schemas.internal import (
    GeneratedAttemptClip,
    SelectedInterval,
)


FFMPEG_INPUT_PATCH_PATH = (
    "ai_judo_coach.attempt_clip_generation."
    "clip_extraction.ffmpeg.input"
)


def test_extract_final_clips_extracts_all_selected_intervals(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    output_directory = (
        tmp_path
        / "generated_clips"
    )

    selected_intervals = [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=1.5,
            end_time_seconds=5.0,
        ),
        SelectedInterval(
            clip_id="1",
            start_time_seconds=8.0,
            end_time_seconds=14.5,
        ),
    ]

    expected_output_paths = [
        output_directory / "attempt_000.mp4",
        output_directory / "attempt_001.mp4",
    ]

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    input_stream_mock = (
        ffmpeg_input_mock.return_value
    )
    output_stream_mock = (
        input_stream_mock.output.return_value
    )
    overwrite_stream_mock = (
        output_stream_mock
        .overwrite_output
        .return_value
    )

    output_paths = iter(expected_output_paths)

    def create_expected_output(**kwargs) -> None:
        del kwargs
        next(output_paths).touch()

    overwrite_stream_mock.run.side_effect = (
        create_expected_output
    )

    result = extract_final_clips(
        selected_intervals=selected_intervals,
        temporary_output_dir_path=str(
            output_directory
        ),
        clip_naming_pattern="attempt_",
        source_video_path=str(source_video_path),
        desired_fps=30.0,
    )

    assert result == [
        GeneratedAttemptClip(
            clip_id="0",
            start_time_seconds=1.5,
            end_time_seconds=5.0,
            file_path=str(
                expected_output_paths[0]
            ),
        ),
        GeneratedAttemptClip(
            clip_id="1",
            start_time_seconds=8.0,
            end_time_seconds=14.5,
            file_path=str(
                expected_output_paths[1]
            ),
        ),
    ]

    assert all(
        output_path.is_file()
        for output_path in expected_output_paths
    )

    assert ffmpeg_input_mock.call_args_list == [
        call(
            str(source_video_path),
            ss=1.5,
        ),
        call(
            str(source_video_path),
            ss=8.0,
        ),
    ]

    assert input_stream_mock.output.call_args_list == [
        call(
            str(expected_output_paths[0]),
            t=3.5,
            vcodec="libx264",
            preset="fast",
            an=None,
            r=30.0,
        ),
        call(
            str(expected_output_paths[1]),
            t=6.5,
            vcodec="libx264",
            preset="fast",
            an=None,
            r=30.0,
        ),
    ]

    assert (
        output_stream_mock
        .overwrite_output
        .call_count
        == 2
    )

    assert overwrite_stream_mock.run.call_args_list == [
        call(
            capture_stdout=True,
            capture_stderr=True,
        ),
        call(
            capture_stdout=True,
            capture_stderr=True,
        ),
    ]


def test_extract_final_clips_creates_nested_output_directory(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    output_directory = (
        tmp_path
        / "job"
        / "generated_clips"
    )

    expected_output_path = (
        output_directory
        / "attempt_007.mp4"
    )

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    run_mock = (
        ffmpeg_input_mock
        .return_value
        .output
        .return_value
        .overwrite_output
        .return_value
        .run
    )

    run_mock.side_effect = (
        lambda **kwargs: (
            expected_output_path.touch()
        )
    )

    result = extract_final_clips(
        selected_intervals=[
            SelectedInterval(
                clip_id="7",
                start_time_seconds=2.0,
                end_time_seconds=4.0,
            ),
        ],
        temporary_output_dir_path=str(
            output_directory
        ),
        clip_naming_pattern="attempt_",
        source_video_path=str(source_video_path),
        desired_fps=30.0,
    )

    assert output_directory.is_dir()
    assert expected_output_path.is_file()
    assert result[0].file_path == str(
        expected_output_path
    )


def test_extract_final_clips_returns_empty_list_for_no_intervals(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    output_directory = (
        tmp_path
        / "generated_clips"
    )

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    result = extract_final_clips(
        selected_intervals=[],
        temporary_output_dir_path=str(
            output_directory
        ),
        clip_naming_pattern="attempt_",
        source_video_path=str(source_video_path),
        desired_fps=30.0,
    )

    assert result == []
    assert output_directory.is_dir()
    ffmpeg_input_mock.assert_not_called()


def test_extract_final_clips_raises_when_source_video_does_not_exist(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = (
        tmp_path
        / "missing.mp4"
    )

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    with pytest.raises(
        FileNotFoundError,
        match="Source video does not exist",
    ):
        extract_final_clips(
            selected_intervals=[],
            temporary_output_dir_path=str(
                tmp_path / "generated_clips"
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=30.0,
        )

    ffmpeg_input_mock.assert_not_called()


@pytest.mark.parametrize(
    "desired_fps",
    [
        0.0,
        -1.0,
    ],
)
def test_extract_final_clips_rejects_non_positive_fps(
    tmp_path: Path,
    mocker,
    desired_fps: float,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    with pytest.raises(
        ValueError,
        match=(
            "desired_fps must be greater than zero"
        ),
    ):
        extract_final_clips(
            selected_intervals=[],
            temporary_output_dir_path=str(
                tmp_path / "generated_clips"
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=desired_fps,
        )

    ffmpeg_input_mock.assert_not_called()


@pytest.mark.parametrize(
    (
        "start_time_seconds",
        "end_time_seconds",
    ),
    [
        (4.0, 4.0),
        (5.0, 4.0),
    ],
)
def test_extract_final_clips_rejects_non_positive_interval_duration(
    tmp_path: Path,
    mocker,
    start_time_seconds: float,
    end_time_seconds: float,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    with pytest.raises(
        ValueError,
        match=(
            "Selected interval duration must be "
            "greater than zero"
        ),
    ):
        extract_final_clips(
            selected_intervals=[
                SelectedInterval(
                    clip_id="0",
                    start_time_seconds=(
                        start_time_seconds
                    ),
                    end_time_seconds=(
                        end_time_seconds
                    ),
                ),
            ],
            temporary_output_dir_path=str(
                tmp_path / "generated_clips"
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=30.0,
        )

    ffmpeg_input_mock.assert_not_called()


def test_extract_final_clips_rejects_non_numeric_clip_id(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    with pytest.raises(
        ValueError,
        match="clip_id must contain a numeric value",
    ):
        extract_final_clips(
            selected_intervals=[
                SelectedInterval(
                    clip_id="invalid",
                    start_time_seconds=1.0,
                    end_time_seconds=2.0,
                ),
            ],
            temporary_output_dir_path=str(
                tmp_path / "generated_clips"
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=30.0,
        )

    ffmpeg_input_mock.assert_not_called()


def test_extract_final_clips_removes_partial_file_when_ffmpeg_fails(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    output_directory = (
        tmp_path
        / "generated_clips"
    )
    expected_output_path = (
        output_directory
        / "attempt_003.mp4"
    )

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    run_mock = (
        ffmpeg_input_mock
        .return_value
        .output
        .return_value
        .overwrite_output
        .return_value
        .run
    )

    ffmpeg_error = ffmpeg.Error(
        "extract",
        b"stdout",
        b"FFmpeg extraction failed",
    )

    def fail_after_creating_partial_file(
        **kwargs,
    ) -> None:
        del kwargs
        expected_output_path.touch()
        raise ffmpeg_error

    run_mock.side_effect = (
        fail_after_creating_partial_file
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Unable to extract attempt clip 3: "
            "FFmpeg extraction failed"
        ),
    ) as exception_info:
        extract_final_clips(
            selected_intervals=[
                SelectedInterval(
                    clip_id="3",
                    start_time_seconds=2.0,
                    end_time_seconds=5.0,
                ),
            ],
            temporary_output_dir_path=str(
                output_directory
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=30.0,
        )

    assert not expected_output_path.exists()
    assert (
        exception_info.value.__cause__
        is ffmpeg_error
    )


def test_extract_final_clips_uses_error_string_when_stderr_is_missing(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    run_mock = (
        ffmpeg_input_mock
        .return_value
        .output
        .return_value
        .overwrite_output
        .return_value
        .run
    )

    ffmpeg_error = ffmpeg.Error(
        "extract",
        b"stdout",
        None,
    )
    run_mock.side_effect = ffmpeg_error

    with pytest.raises(
        RuntimeError,
        match="Unable to extract attempt clip 4",
    ) as exception_info:
        extract_final_clips(
            selected_intervals=[
                SelectedInterval(
                    clip_id="4",
                    start_time_seconds=2.0,
                    end_time_seconds=5.0,
                ),
            ],
            temporary_output_dir_path=str(
                tmp_path / "generated_clips"
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=30.0,
        )

    assert (
        exception_info.value.__cause__
        is ffmpeg_error
    )


def test_extract_final_clips_raises_when_ffmpeg_creates_no_file(
    tmp_path: Path,
    mocker,
) -> None:
    source_video_path = tmp_path / "source.mp4"
    source_video_path.touch()

    output_directory = (
        tmp_path
        / "generated_clips"
    )
    expected_output_path = (
        output_directory
        / "attempt_005.mp4"
    )

    ffmpeg_input_mock = mocker.patch(
        FFMPEG_INPUT_PATCH_PATH
    )

    run_mock = (
        ffmpeg_input_mock
        .return_value
        .output
        .return_value
        .overwrite_output
        .return_value
        .run
    )
    run_mock.return_value = None

    with pytest.raises(
        RuntimeError,
        match=(
            "FFmpeg completed without creating "
            "the expected clip"
        ),
    ):
        extract_final_clips(
            selected_intervals=[
                SelectedInterval(
                    clip_id="5",
                    start_time_seconds=2.0,
                    end_time_seconds=5.0,
                ),
            ],
            temporary_output_dir_path=str(
                output_directory
            ),
            clip_naming_pattern="attempt_",
            source_video_path=str(
                source_video_path
            ),
            desired_fps=30.0,
        )

    assert not expected_output_path.exists()
