"""Takes a clip as list of numpy arrays all the way to its LSTM classification."""

import numpy as np
from ultralytics import YOLO

from v1_clip_classification_model.inference import (
    JudoClipClassifier,
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
) -> ClipProcessingResult:
    """
    Takes a clip as a list of NumPy arrays all the way to its
    JudoClipClassifier classification.

    The higher-level orchestrator handles looping over clips and
    constructs the model instances once so they can be reused.
    """

    # Mini pipeline taking a single clip through YOLO, player detection,
    # LSTM input construction, and LSTM classification.
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
