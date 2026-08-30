import math
from pathlib import Path

import numpy as np
import pytest

from v1_clip_classification_model.inference import (
    JudoClipClassifier,
)

from ai_judo_coach.config import (
    JUDO_CLIPPER_MODEL_DIRECTORY,
)
from ai_judo_coach.inference.inference_schemas import (
    ClipClassificationResult,
)
from ai_judo_coach.inference.judo_clip_classifier_handler import (
    construct_classifier,
    predict,
)


pytestmark = pytest.mark.integration


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _get_classifier_release_directory() -> Path:
    """Return the configured classifier release directory."""

    release_directory = Path(
        JUDO_CLIPPER_MODEL_DIRECTORY
    )

    if not release_directory.is_absolute():
        release_directory = (
            PROJECT_ROOT
            / release_directory
        )

    return release_directory.resolve()


@pytest.fixture(scope="module")
def released_classifier() -> JudoClipClassifier:
    """Load the real released classifier on CPU."""

    release_directory = (
        _get_classifier_release_directory()
    )

    if not release_directory.is_dir():
        pytest.skip(
            "Classifier release directory is unavailable: "
            f"{release_directory}"
        )

    metadata_path = (
        release_directory
        / "model_metadata.yaml"
    )
    weights_path = (
        release_directory
        / "model_weights.pt"
    )

    if not metadata_path.is_file():
        pytest.skip(
            "Classifier metadata file is unavailable: "
            f"{metadata_path}"
        )

    if not weights_path.is_file():
        pytest.skip(
            "Classifier weights file is unavailable: "
            f"{weights_path}"
        )

    return construct_classifier(
        classifier_release_directory=(
            release_directory
        ),
        classifier_device="cpu",
    )


def test_released_classifier_returns_valid_prediction(
    released_classifier: JudoClipClassifier,
) -> None:
    model_input = np.linspace(
        start=0.0,
        stop=1.0,
        num=210 * 68,
        dtype=np.float32,
    ).reshape(
        210,
        68,
    )

    result = predict(
        classifier=released_classifier,
        input_array=model_input,
    )

    assert isinstance(
        result,
        ClipClassificationResult,
    )

    assert math.isfinite(result.logit)
    assert math.isfinite(result.probability)
    assert 0.0 <= result.probability <= 1.0

    assert result.prediction in (0, 1)
    assert result.class_name in (
        "no_attempt",
        "attempt",
    )

    assert result.threshold == pytest.approx(
        0.55
    )

    assert result.prediction == int(
        result.probability
        >= result.threshold
    )

    expected_class_name = (
        "attempt"
        if result.prediction == 1
        else "no_attempt"
    )

    assert (
        result.class_name
        == expected_class_name
    )

    expected_probability = (
        1.0
        / (
            1.0
            + math.exp(-result.logit)
        )
    )

    assert result.probability == pytest.approx(
        expected_probability,
        abs=1e-6,
    )


def test_released_classifier_prediction_is_deterministic_on_cpu(
    released_classifier: JudoClipClassifier,
) -> None:
    random_generator = np.random.default_rng(
        seed=42
    )

    model_input = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(210, 68),
    ).astype(
        np.float32
    )

    first_result = predict(
        classifier=released_classifier,
        input_array=model_input,
    )
    second_result = predict(
        classifier=released_classifier,
        input_array=model_input,
    )

    assert second_result.logit == pytest.approx(
        first_result.logit,
        abs=1e-7,
    )
    assert second_result.probability == pytest.approx(
        first_result.probability,
        abs=1e-7,
    )
    assert (
        second_result.prediction
        == first_result.prediction
    )
    assert (
        second_result.class_name
        == first_result.class_name
    )
    assert second_result.threshold == pytest.approx(
        first_result.threshold,
        abs=1e-7,
    )


def test_released_classifier_handles_float64_and_non_finite_values(
    released_classifier: JudoClipClassifier,
) -> None:
    model_input = np.zeros(
        (210, 68),
        dtype=np.float64,
    )

    model_input[0, 0] = np.nan
    model_input[1, 1] = np.inf
    model_input[2, 2] = -np.inf

    result = predict(
        classifier=released_classifier,
        input_array=model_input,
    )

    assert isinstance(
        result,
        ClipClassificationResult,
    )
    assert math.isfinite(result.logit)
    assert math.isfinite(result.probability)
    assert 0.0 <= result.probability <= 1.0
    assert result.prediction in (0, 1)
    assert result.class_name in (
        "no_attempt",
        "attempt",
    )
    assert result.threshold == pytest.approx(
        0.55
    )
