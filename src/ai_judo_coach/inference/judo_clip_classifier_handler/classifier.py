"""Interface with the released clip-classification model."""

from pathlib import Path

import numpy as np

from v1_clip_classification_model.inference import (
    JudoClipClassifier,
)

from ai_judo_coach.exceptions import (
    ClassifierLoadingError,
    InvalidClassifierInputError,
)
from ai_judo_coach.inference.inference_schemas import (
    ClipClassificationResult,
)


def construct_classifier(
    classifier_release_directory: str | Path,
    classifier_device: str,
) -> JudoClipClassifier:
    """Construct an inference-ready clip classifier."""

    try:
        return JudoClipClassifier.from_release(
            release_directory=classifier_release_directory,
            device=classifier_device,
        )
    except Exception as error:
        raise ClassifierLoadingError(
            "Classifier model failed to load"
        ) from error


def predict(
    classifier: JudoClipClassifier,
    input_array: np.ndarray,
) -> ClipClassificationResult:
    """Classify one complete LSTM input sequence."""

    try:
        model_result = classifier.predict(
            model_input=input_array,
        )
    except (TypeError, ValueError) as error:
        raise InvalidClassifierInputError(
            "Classifier input must be a NumPy array with shape "
            "(210, 68)"
        ) from error

    return ClipClassificationResult(
        logit=model_result.logit,
        probability=model_result.probability,
        prediction=model_result.prediction,
        class_name=model_result.class_name,
        threshold=model_result.threshold,
    )
