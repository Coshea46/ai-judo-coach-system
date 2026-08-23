import numpy as np
import pytest

from ai_judo_coach.inference.clip_inference import (
    process_clip,
)
from ai_judo_coach.inference.inference_schemas import (
    ClipClassificationResult,
)
from ai_judo_coach.inference.player_detection import (
    PoseSequenceQualityReport,
)
from ai_judo_coach.schemas.internal import (
    ClipProcessingResult,
)


CLIP_INFERENCE_MODULE_PATH = (
    "ai_judo_coach.inference.clip_inference"
)


def _create_quality_report(
    accepted: bool,
) -> PoseSequenceQualityReport:
    """Create a pose-sequence quality report for testing."""

    return PoseSequenceQualityReport(
        accepted=accepted,
        rejection_reasons=(
            ()
            if accepted
            else (
                "player_a_unusable_frame_fraction_exceeded",
            )
        ),
        player_a_unusable_frame_fraction=(
            0.0
            if accepted
            else 0.5
        ),
        player_b_unusable_frame_fraction=0.0,
        player_a_longest_unusable_gap=(
            0
            if accepted
            else 20
        ),
        player_b_longest_unusable_gap=0,
        both_players_unusable_fraction=0.0,
    )


def test_process_clip_runs_complete_inference_pipeline(
    mocker,
) -> None:
    clip_as_numpy = [
        np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        ),
        np.ones(
            (720, 1280, 3),
            dtype=np.uint8,
        ),
    ]

    yolo_model = mocker.Mock()
    judo_clip_classifier = mocker.Mock()

    yolo_results = iter(
        [
            mocker.Mock(),
            mocker.Mock(),
        ]
    )
    clip_detections = mocker.Mock()
    player_pose_sequences = mocker.Mock()
    quality_report = _create_quality_report(
        accepted=True,
    )

    classifier_input_array = np.zeros(
        (210, 68),
        dtype=np.float32,
    )

    classification_result = ClipClassificationResult(
        logit=1.25,
        probability=0.78,
        prediction=1,
        class_name="attempt",
        threshold=0.55,
    )

    track_video_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "track_video",
        return_value=yolo_results,
    )
    collect_detections_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "collect_clip_detections",
        return_value=clip_detections,
    )
    detect_players_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "detect_players",
        return_value=(
            player_pose_sequences,
            quality_report,
        ),
    )
    build_input_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "build_lstm_input_array",
        return_value=classifier_input_array,
    )
    predict_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "predict",
        return_value=classification_result,
    )

    result = process_clip(
        clip_as_numpy=clip_as_numpy,
        clip_id="clip_007",
        yolo_model=yolo_model,
        yolo_tracker_path="config/bytetrack.yaml",
        yolo_device="cuda:0",
        judo_clip_classifier=(
            judo_clip_classifier
        ),
    )

    assert result == ClipProcessingResult(
        clip_id="clip_007",
        contains_throw_attempt=True,
        attempt_probability=0.78,
        predicted_class_name="attempt",
    )

    track_video_mock.assert_called_once_with(
        yolo_model=yolo_model,
        tracker_path="config/bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device="cuda:0",
    )

    collect_detections_mock.assert_called_once_with(
        clip_id="clip_007",
        yolo_clip_output=yolo_results,
    )

    detect_players_mock.assert_called_once_with(
        clip_detections=clip_detections,
    )

    build_input_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            player_pose_sequences
        ),
        pose_sequence_quality_report=(
            quality_report
        ),
    )

    predict_mock.assert_called_once_with(
        classifier=judo_clip_classifier,
        input_array=classifier_input_array,
    )


@pytest.mark.parametrize(
    (
        "prediction",
        "class_name",
        "expected_contains_throw_attempt",
    ),
    [
        (
            1,
            "no_attempt",
            True,
        ),
        (
            0,
            "attempt",
            False,
        ),
    ],
)
def test_process_clip_uses_numeric_prediction_for_attempt_decision(
    mocker,
    prediction: int,
    class_name: str,
    expected_contains_throw_attempt: bool,
) -> None:
    quality_report = _create_quality_report(
        accepted=True,
    )
    player_pose_sequences = mocker.Mock()

    classifier_input_array = np.zeros(
        (210, 68),
        dtype=np.float32,
    )

    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "track_video",
        return_value=iter(()),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "collect_clip_detections",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "detect_players",
        return_value=(
            player_pose_sequences,
            quality_report,
        ),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "build_lstm_input_array",
        return_value=classifier_input_array,
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "predict",
        return_value=ClipClassificationResult(
            logit=0.75,
            probability=0.68,
            prediction=prediction,
            class_name=class_name,
            threshold=0.55,
        ),
    )

    result = process_clip(
        clip_as_numpy=[],
        clip_id="clip_0",
        yolo_model=mocker.Mock(),
        yolo_tracker_path="bytetrack.yaml",
        yolo_device="cpu",
        judo_clip_classifier=mocker.Mock(),
    )

    assert (
        result.contains_throw_attempt
        is expected_contains_throw_attempt
    )
    assert result.attempt_probability == pytest.approx(
        0.68
    )
    assert result.predicted_class_name == class_name


def test_process_clip_returns_no_attempt_when_quality_is_rejected(
    mocker,
) -> None:
    clip_as_numpy = [
        np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        ),
    ]

    yolo_model = mocker.Mock()
    judo_clip_classifier = mocker.Mock()

    yolo_results = iter(
        [
            mocker.Mock(),
        ]
    )
    clip_detections = mocker.Mock()
    player_pose_sequences = mocker.Mock()

    rejected_quality_report = (
        _create_quality_report(
            accepted=False,
        )
    )

    track_video_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "track_video",
        return_value=yolo_results,
    )
    collect_detections_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "collect_clip_detections",
        return_value=clip_detections,
    )
    detect_players_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "detect_players",
        return_value=(
            player_pose_sequences,
            rejected_quality_report,
        ),
    )
    build_input_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "build_lstm_input_array",
        return_value=None,
    )
    predict_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "predict",
    )

    result = process_clip(
        clip_as_numpy=clip_as_numpy,
        clip_id="clip_rejected",
        yolo_model=yolo_model,
        yolo_tracker_path="bytetrack.yaml",
        yolo_device="cpu",
        judo_clip_classifier=(
            judo_clip_classifier
        ),
    )

    assert result == ClipProcessingResult(
        clip_id="clip_rejected",
        contains_throw_attempt=False,
        attempt_probability=0.0,
        predicted_class_name="no_attempt",
    )

    track_video_mock.assert_called_once_with(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=clip_as_numpy,
        compute_device="cpu",
    )
    collect_detections_mock.assert_called_once_with(
        clip_id="clip_rejected",
        yolo_clip_output=yolo_results,
    )
    detect_players_mock.assert_called_once_with(
        clip_detections=clip_detections,
    )
    build_input_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            player_pose_sequences
        ),
        pose_sequence_quality_report=(
            rejected_quality_report
        ),
    )

    predict_mock.assert_not_called()
    judo_clip_classifier.predict.assert_not_called()


def test_process_clip_returns_no_attempt_when_lstm_input_cannot_be_built(
    mocker,
) -> None:
    quality_report = _create_quality_report(
        accepted=True,
    )
    player_pose_sequences = mocker.Mock()

    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "track_video",
        return_value=iter(()),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "collect_clip_detections",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "detect_players",
        return_value=(
            player_pose_sequences,
            quality_report,
        ),
    )
    build_input_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "build_lstm_input_array",
        return_value=None,
    )
    predict_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "predict",
    )

    result = process_clip(
        clip_as_numpy=[],
        clip_id="invalid_shape_clip",
        yolo_model=mocker.Mock(),
        yolo_tracker_path="bytetrack.yaml",
        yolo_device="cpu",
        judo_clip_classifier=mocker.Mock(),
    )

    assert result == ClipProcessingResult(
        clip_id="invalid_shape_clip",
        contains_throw_attempt=False,
        attempt_probability=0.0,
        predicted_class_name="no_attempt",
    )

    build_input_mock.assert_called_once_with(
        clip_player_pose_sequences=(
            player_pose_sequences
        ),
        pose_sequence_quality_report=(
            quality_report
        ),
    )
    predict_mock.assert_not_called()


def test_process_clip_preserves_negative_classification_probability(
    mocker,
) -> None:
    quality_report = _create_quality_report(
        accepted=True,
    )

    classifier_input_array = np.zeros(
        (210, 68),
        dtype=np.float32,
    )

    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "track_video",
        return_value=iter(()),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "collect_clip_detections",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "detect_players",
        return_value=(
            mocker.Mock(),
            quality_report,
        ),
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "build_lstm_input_array",
        return_value=classifier_input_array,
    )
    mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "predict",
        return_value=ClipClassificationResult(
            logit=-0.8,
            probability=0.31,
            prediction=0,
            class_name="no_attempt",
            threshold=0.55,
        ),
    )

    result = process_clip(
        clip_as_numpy=[],
        clip_id="negative_clip",
        yolo_model=mocker.Mock(),
        yolo_tracker_path="bytetrack.yaml",
        yolo_device="cpu",
        judo_clip_classifier=mocker.Mock(),
    )

    assert result == ClipProcessingResult(
        clip_id="negative_clip",
        contains_throw_attempt=False,
        attempt_probability=0.31,
        predicted_class_name="no_attempt",
    )


def test_process_clip_propagates_yolo_tracking_failure_and_stops_pipeline(
    mocker,
) -> None:
    tracking_error = RuntimeError(
        "YOLO tracking failed"
    )

    track_video_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "track_video",
        side_effect=tracking_error,
    )
    collect_detections_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "collect_clip_detections",
    )
    detect_players_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "detect_players",
    )
    build_input_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "build_lstm_input_array",
    )
    predict_mock = mocker.patch(
        f"{CLIP_INFERENCE_MODULE_PATH}."
        "predict",
    )

    yolo_model = mocker.Mock()
    judo_clip_classifier = mocker.Mock()

    with pytest.raises(
        RuntimeError,
        match="YOLO tracking failed",
    ) as exception_info:
        process_clip(
            clip_as_numpy=[],
            clip_id="clip_0",
            yolo_model=yolo_model,
            yolo_tracker_path="bytetrack.yaml",
            yolo_device="cpu",
            judo_clip_classifier=(
                judo_clip_classifier
            ),
        )

    assert exception_info.value is tracking_error

    track_video_mock.assert_called_once_with(
        yolo_model=yolo_model,
        tracker_path="bytetrack.yaml",
        clip_as_numpy=[],
        compute_device="cpu",
    )
    collect_detections_mock.assert_not_called()
    detect_players_mock.assert_not_called()
    build_input_mock.assert_not_called()
    predict_mock.assert_not_called()
