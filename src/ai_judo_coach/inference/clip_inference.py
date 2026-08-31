"""Takes a clip as list of numpy arrays all the way to its LSTM classification."""

import numpy as np
from ultralytics import YOLO

from v1_clip_classification_model.inference import (
    JudoClipClassifier,
)

from ai_judo_coach.inference.inference_schemas import (
    ClipDetections,
    FrameDetections,
)
from ai_judo_coach.schemas.internal import (
    ClipProcessingResult,
)

from .judo_clip_classifier_handler import (
    build_lstm_input_array,
    predict,
)
from .player_detection import (
    detect_players,
)
from .yolo_feeder import (
    collect_cached_tracked_clip_detections,
    collect_clip_detections,
    track_video,
)


# Shouldn't have config.py as a dependency in this file.
# Only the top-level orchestrator should touch config.py.
def process_clip(
    clip_as_numpy: list[np.ndarray],
    clip_id: str,
    yolo_model: YOLO,
    yolo_tracker_path: str,
    yolo_device: str,
    judo_clip_classifier: JudoClipClassifier,
    absolute_frame_indices: list[int] | None = None,
    pose_detection_frame_indices: list[int] | None = None,
    pose_detection_cache: dict[int, FrameDetections] | None = None,
) -> ClipProcessingResult:
    """
    Takes a clip as a list of NumPy arrays all the way to its
    JudoClipClassifier classification.

    The higher-level orchestrator handles looping over clips and
    constructs the model instances once so they can be reused.

    When absolute frame indices, pose-detection frame indices and a
    shared pose-detection cache are supplied, YOLO pose inference is
    reused across overlapping clips. ByteTrack is still reset and
    rerun independently for every clip.
    """

    cache_arguments = (
        absolute_frame_indices,
        pose_detection_frame_indices,
        pose_detection_cache,
    )

    cache_arguments_are_incomplete = (
        any(
            argument is not None
            for argument in cache_arguments
        )
        and not all(
            argument is not None
            for argument in cache_arguments
        )
    )

    if cache_arguments_are_incomplete:
        raise ValueError(
            "absolute_frame_indices, "
            "pose_detection_frame_indices and "
            "pose_detection_cache must either all be supplied "
            "or all be omitted"
        )

    if (
        absolute_frame_indices is not None
        and pose_detection_frame_indices is not None
        and pose_detection_cache is not None
    ):
        clip_detections = (
            collect_cached_tracked_clip_detections(
                yolo_model=yolo_model,
                tracker_path=yolo_tracker_path,
                clip_as_numpy=clip_as_numpy,
                absolute_frame_indices=(
                    absolute_frame_indices
                ),
                pose_detection_frame_indices=(
                    pose_detection_frame_indices
                ),
                compute_device=yolo_device,
                pose_detection_cache=(
                    pose_detection_cache
                ),
                clip_id=clip_id,
            )
        )

    else:
        # Mini pipeline taking a single clip through YOLO and
        # conversion into the project's detection schemas.
        yolo_results = track_video(
            yolo_model=yolo_model,
            tracker_path=yolo_tracker_path,
            clip_as_numpy=clip_as_numpy,
            compute_device=yolo_device,
        )

        clip_detections = collect_clip_detections(
            clip_id=clip_id,
            yolo_clip_output=yolo_results,
        )

    return _process_clip_detections(
        clip_detections=clip_detections,
        clip_id=clip_id,
        judo_clip_classifier=judo_clip_classifier,
    )


def _process_clip_detections(
    clip_detections: ClipDetections,
    clip_id: str,
    judo_clip_classifier: JudoClipClassifier,
) -> ClipProcessingResult:
    """
    Run player assignment and classification on clip detections.

    Both the original YOLO tracking path and the cached pose-detection
    path use this same downstream processing.
    """

    player_pose_sequences, quality_report = detect_players(
        clip_detections=clip_detections,
    )

    classifier_input_array = build_lstm_input_array(
        clip_player_pose_sequences=player_pose_sequences,
        pose_sequence_quality_report=quality_report,
    )

    if classifier_input_array is None:
        return ClipProcessingResult(
            clip_id=clip_id,
            contains_throw_attempt=False,
            attempt_probability=0.0,
            predicted_class_name="no_attempt",
        )

    # Now run through the Judo clip-classifier model.
    clip_classification_result = predict(
        classifier=judo_clip_classifier,
        input_array=classifier_input_array,
    )

    # Package as type ClipProcessingResult.
    contains_throw_attempt = (
        clip_classification_result.prediction == 1
    )

    return ClipProcessingResult(
        clip_id=clip_id,
        contains_throw_attempt=contains_throw_attempt,
        attempt_probability=(
            clip_classification_result.probability
        ),
        predicted_class_name=(
            clip_classification_result.class_name
        ),
    )
