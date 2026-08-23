from pathlib import Path
from unittest.mock import call

import numpy as np
import pytest

import ai_judo_coach.pipeline.orchestrator as orchestrator
from ai_judo_coach.schemas.internal import (
    ClipProcessingResult,
    DetectedAttemptWindow,
    GeneratedAttemptClip,
    InitialClipWindow,
    SelectedInterval,
)


ORCHESTRATOR_MODULE_PATH = (
    "ai_judo_coach.pipeline.orchestrator"
)


def test_run_pipeline_processes_windows_and_extracts_positive_intervals(
    mocker,
) -> None:
    input_video_path = "/input/source.mp4"
    temporary_output_directory = "/tmp/job_123"
    cleansed_video_path = (
        "/tmp/job_123/input_cleanse/cleansed_input.mp4"
    )
    cleansed_video_duration = 30.0

    clip_windows = [
        InitialClipWindow(
            start_time=0.0,
            end_time=7.0,
            window_id=0,
        ),
        InitialClipWindow(
            start_time=3.0,
            end_time=10.0,
            window_id=1,
        ),
        InitialClipWindow(
            start_time=6.0,
            end_time=13.0,
            window_id=2,
        ),
    ]

    clip_frames = [
        [
            np.zeros(
                (32, 32, 3),
                dtype=np.uint8,
            ),
        ],
        [
            np.ones(
                (32, 32, 3),
                dtype=np.uint8,
            ),
        ],
        [
            np.full(
                (32, 32, 3),
                2,
                dtype=np.uint8,
            ),
        ],
    ]

    clip_processing_results = [
        ClipProcessingResult(
            clip_id="0",
            contains_throw_attempt=True,
            attempt_probability=0.81,
            predicted_class_name="attempt",
        ),
        ClipProcessingResult(
            clip_id="1",
            contains_throw_attempt=False,
            attempt_probability=0.25,
            predicted_class_name="no_attempt",
        ),
        ClipProcessingResult(
            clip_id="2",
            contains_throw_attempt=True,
            attempt_probability=0.73,
            predicted_class_name="attempt",
        ),
    ]

    selected_intervals = [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=0.0,
            end_time_seconds=13.0,
        ),
    ]

    expected_generated_clips = [
        GeneratedAttemptClip(
            clip_id="0",
            start_time_seconds=0.0,
            end_time_seconds=13.0,
            file_path=(
                "/tmp/job_123/generated_clips/"
                "attempt_000.mp4"
            ),
        ),
    ]

    yolo_model = mocker.Mock()
    classifier_model = mocker.Mock()

    cleanse_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "cleanse_input_video",
        return_value=(
            cleansed_video_path,
            cleansed_video_duration,
        ),
    )
    compute_windows_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "compute_initial_clip_windows",
        return_value=iter(clip_windows),
    )
    resolve_device_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "resolve_yolo_device",
        return_value="cuda:0",
    )
    load_yolo_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "load_yolo_model",
        return_value=yolo_model,
    )
    construct_classifier_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "construct_classifier",
        return_value=classifier_model,
    )
    extract_frames_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_frames_from_initial_window",
        side_effect=clip_frames,
    )
    process_clip_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "process_clip",
        side_effect=clip_processing_results,
    )
    select_intervals_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "select_new_intervals",
        return_value=selected_intervals,
    )
    extract_final_clips_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_final_clips",
        return_value=expected_generated_clips,
    )

    result = orchestrator.run_pipeline(
        input_video_path=input_video_path,
        temporary_output_directory=(
            temporary_output_directory
        ),
    )

    assert result == expected_generated_clips
    assert result is expected_generated_clips

    cleanse_mock.assert_called_once_with(
        input_video_path=input_video_path,
        output_directory=temporary_output_directory,
    )

    compute_windows_mock.assert_called_once_with(
        input_video_path=cleansed_video_path,
        individual_window_duration=float(
            orchestrator.CLIP_DURATION_SEC
        ),
        stride=float(
            orchestrator.CLIP_STRIDE_SEC
        ),
    )

    resolve_device_mock.assert_called_once_with(
        requested_device=orchestrator.YOLO_DEVICE,
    )

    load_yolo_mock.assert_called_once_with(
        yolo_model_path=(
            orchestrator.YOLO_MODEL_WEIGHTS
        ),
    )

    construct_classifier_mock.assert_called_once_with(
        classifier_release_directory=(
            orchestrator.JUDO_CLIPPER_MODEL_DIRECTORY
        ),
        classifier_device=(
            orchestrator.CLASSIFIER_DEVICE
        ),
    )

    assert extract_frames_mock.call_args_list == [
        call(
            source_video_path=cleansed_video_path,
            window=clip_windows[0],
            video_fps=float(
                orchestrator.TARGET_FPS
            ),
            device=orchestrator.DECORD_TARGET_DEVICE,
        ),
        call(
            source_video_path=cleansed_video_path,
            window=clip_windows[1],
            video_fps=float(
                orchestrator.TARGET_FPS
            ),
            device=orchestrator.DECORD_TARGET_DEVICE,
        ),
        call(
            source_video_path=cleansed_video_path,
            window=clip_windows[2],
            video_fps=float(
                orchestrator.TARGET_FPS
            ),
            device=orchestrator.DECORD_TARGET_DEVICE,
        ),
    ]

    assert process_clip_mock.call_count == 3

    for clip_index, process_call in enumerate(
        process_clip_mock.call_args_list
    ):
        call_arguments = process_call.kwargs

        assert (
            call_arguments["clip_as_numpy"]
            is clip_frames[clip_index]
        )
        assert call_arguments["clip_id"] == str(
            clip_windows[clip_index].window_id
        )
        assert (
            call_arguments["yolo_model"]
            is yolo_model
        )
        assert (
            call_arguments["yolo_tracker_path"]
            == orchestrator.BYTETRACK_CONFIG_PATH
        )
        assert (
            call_arguments["yolo_device"]
            == "cuda:0"
        )
        assert (
            call_arguments["judo_clip_classifier"]
            is classifier_model
        )

    select_intervals_mock.assert_called_once_with(
        surviving_initial_windows=[
            DetectedAttemptWindow(
                window=clip_windows[0],
                attempt_probability=0.81,
            ),
            DetectedAttemptWindow(
                window=clip_windows[2],
                attempt_probability=0.73,
            ),
        ],
        source_video_duration=(
            cleansed_video_duration
        ),
        max_new_interval_duration=(
            orchestrator
            .MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC
        ),
        max_intervals_per_video=(
            orchestrator.MAX_GENERATED_ATTEMPT_CLIPS
        ),
    )

    expected_output_directory = str(
        Path(temporary_output_directory)
        / "generated_clips"
    )

    extract_final_clips_mock.assert_called_once_with(
        selected_intervals=selected_intervals,
        temporary_output_dir_path=(
            expected_output_directory
        ),
        clip_naming_pattern=(
            orchestrator.OUTPUT_CLIP_NAMING_PATTERN
        ),
        source_video_path=cleansed_video_path,
        desired_fps=orchestrator.TARGET_FPS,
    )


def test_run_pipeline_returns_empty_list_when_no_attempts_are_detected(
    mocker,
) -> None:
    input_video_path = "/input/source.mp4"
    temporary_output_directory = "/tmp/job_123"
    cleansed_video_path = (
        "/tmp/job_123/input_cleanse/cleansed_input.mp4"
    )

    clip_windows = [
        InitialClipWindow(
            start_time=0.0,
            end_time=7.0,
            window_id=0,
        ),
        InitialClipWindow(
            start_time=3.0,
            end_time=10.0,
            window_id=1,
        ),
    ]

    clip_frames = [
        [
            np.zeros(
                (16, 16, 3),
                dtype=np.uint8,
            ),
        ],
        [
            np.ones(
                (16, 16, 3),
                dtype=np.uint8,
            ),
        ],
    ]

    clip_processing_results = [
        ClipProcessingResult(
            clip_id="0",
            contains_throw_attempt=False,
            attempt_probability=0.49,
            predicted_class_name="attempt",
        ),
        ClipProcessingResult(
            clip_id="1",
            contains_throw_attempt=False,
            attempt_probability=0.10,
            predicted_class_name="no_attempt",
        ),
    ]

    yolo_model = mocker.Mock()
    classifier_model = mocker.Mock()

    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "cleanse_input_video",
        return_value=(
            cleansed_video_path,
            20.0,
        ),
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "compute_initial_clip_windows",
        return_value=iter(clip_windows),
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "resolve_yolo_device",
        return_value="cpu",
    )
    load_yolo_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "load_yolo_model",
        return_value=yolo_model,
    )
    construct_classifier_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "construct_classifier",
        return_value=classifier_model,
    )
    extract_frames_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_frames_from_initial_window",
        side_effect=clip_frames,
    )
    process_clip_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "process_clip",
        side_effect=clip_processing_results,
    )
    select_intervals_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "select_new_intervals",
    )
    extract_final_clips_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_final_clips",
    )

    result = orchestrator.run_pipeline(
        input_video_path=input_video_path,
        temporary_output_directory=(
            temporary_output_directory
        ),
    )

    assert result == []

    load_yolo_mock.assert_called_once_with(
        yolo_model_path=(
            orchestrator.YOLO_MODEL_WEIGHTS
        ),
    )
    construct_classifier_mock.assert_called_once_with(
        classifier_release_directory=(
            orchestrator.JUDO_CLIPPER_MODEL_DIRECTORY
        ),
        classifier_device=(
            orchestrator.CLASSIFIER_DEVICE
        ),
    )

    assert extract_frames_mock.call_count == 2
    assert process_clip_mock.call_count == 2

    # The pipeline uses contains_throw_attempt rather than the
    # predicted class-name string.
    assert (
        clip_processing_results[0]
        .predicted_class_name
        == "attempt"
    )

    select_intervals_mock.assert_not_called()
    extract_final_clips_mock.assert_not_called()


def test_run_pipeline_returns_empty_list_when_there_are_no_initial_windows(
    mocker,
) -> None:
    cleansed_video_path = (
        "/tmp/job/input_cleanse/cleansed_input.mp4"
    )

    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "cleanse_input_video",
        return_value=(
            cleansed_video_path,
            2.0,
        ),
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "compute_initial_clip_windows",
        return_value=iter(()),
    )
    resolve_device_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "resolve_yolo_device",
        return_value="cpu",
    )
    load_yolo_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "load_yolo_model",
        return_value=mocker.Mock(),
    )
    construct_classifier_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "construct_classifier",
        return_value=mocker.Mock(),
    )
    extract_frames_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_frames_from_initial_window",
    )
    process_clip_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "process_clip",
    )
    select_intervals_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "select_new_intervals",
    )
    extract_final_clips_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_final_clips",
    )

    result = orchestrator.run_pipeline(
        input_video_path="/input/short.mp4",
        temporary_output_directory="/tmp/job",
    )

    assert result == []

    resolve_device_mock.assert_called_once_with(
        requested_device=orchestrator.YOLO_DEVICE,
    )
    load_yolo_mock.assert_called_once()
    construct_classifier_mock.assert_called_once()

    extract_frames_mock.assert_not_called()
    process_clip_mock.assert_not_called()
    select_intervals_mock.assert_not_called()
    extract_final_clips_mock.assert_not_called()


def test_run_pipeline_propagates_cleansing_failure_and_stops_pipeline(
    mocker,
) -> None:
    cleansing_error = RuntimeError(
        "Unable to cleanse input video"
    )

    cleanse_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "cleanse_input_video",
        side_effect=cleansing_error,
    )
    compute_windows_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "compute_initial_clip_windows",
    )
    resolve_device_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "resolve_yolo_device",
    )
    load_yolo_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "load_yolo_model",
    )
    construct_classifier_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "construct_classifier",
    )
    process_clip_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "process_clip",
    )
    select_intervals_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "select_new_intervals",
    )
    extract_final_clips_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_final_clips",
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to cleanse input video",
    ) as exception_info:
        orchestrator.run_pipeline(
            input_video_path="/input/source.mp4",
            temporary_output_directory="/tmp/job",
        )

    assert exception_info.value is cleansing_error

    cleanse_mock.assert_called_once_with(
        input_video_path="/input/source.mp4",
        output_directory="/tmp/job",
    )

    compute_windows_mock.assert_not_called()
    resolve_device_mock.assert_not_called()
    load_yolo_mock.assert_not_called()
    construct_classifier_mock.assert_not_called()
    process_clip_mock.assert_not_called()
    select_intervals_mock.assert_not_called()
    extract_final_clips_mock.assert_not_called()


def test_run_pipeline_propagates_clip_processing_failure_and_does_not_extract_clips(
    mocker,
) -> None:
    cleansed_video_path = (
        "/tmp/job/input_cleanse/cleansed_input.mp4"
    )

    clip_window = InitialClipWindow(
        start_time=0.0,
        end_time=7.0,
        window_id=0,
    )
    clip_frames = [
        np.zeros(
            (16, 16, 3),
            dtype=np.uint8,
        ),
    ]

    processing_error = RuntimeError(
        "Clip processing failed"
    )

    yolo_model = mocker.Mock()
    classifier_model = mocker.Mock()

    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "cleanse_input_video",
        return_value=(
            cleansed_video_path,
            10.0,
        ),
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "compute_initial_clip_windows",
        return_value=iter(
            [clip_window]
        ),
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "resolve_yolo_device",
        return_value="cpu",
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "load_yolo_model",
        return_value=yolo_model,
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "construct_classifier",
        return_value=classifier_model,
    )
    mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_frames_from_initial_window",
        return_value=clip_frames,
    )
    process_clip_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "process_clip",
        side_effect=processing_error,
    )
    select_intervals_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "select_new_intervals",
    )
    extract_final_clips_mock = mocker.patch(
        f"{ORCHESTRATOR_MODULE_PATH}."
        "extract_final_clips",
    )

    with pytest.raises(
        RuntimeError,
        match="Clip processing failed",
    ) as exception_info:
        orchestrator.run_pipeline(
            input_video_path="/input/source.mp4",
            temporary_output_directory="/tmp/job",
        )

    assert exception_info.value is processing_error

    process_clip_mock.assert_called_once_with(
        clip_as_numpy=clip_frames,
        clip_id="0",
        yolo_model=yolo_model,
        yolo_tracker_path=(
            orchestrator.BYTETRACK_CONFIG_PATH
        ),
        yolo_device="cpu",
        judo_clip_classifier=classifier_model,
    )

    select_intervals_mock.assert_not_called()
    extract_final_clips_mock.assert_not_called()
