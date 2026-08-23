from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai_judo_coach.exceptions import (
    ClassifierLoadingError,
    InvalidClassifierInputError,
)
from ai_judo_coach.inference.inference_schemas import (
    ClipClassificationResult,
)
from ai_judo_coach.inference.judo_clip_classifier_handler.classifier import (
    construct_classifier,
    predict,
)


FROM_RELEASE_PATCH_PATH = (
    "ai_judo_coach.inference."
    "judo_clip_classifier_handler.classifier."
    "JudoClipClassifier.from_release"
)


@pytest.mark.parametrize(
    "release_directory",
    [
        "/weights/classifier_release",
        Path("/weights/classifier_release"),
    ],
)
def test_construct_classifier_loads_classifier_from_release(
    mocker,
    release_directory: str | Path,
) -> None:
    expected_classifier = mocker.Mock()

    from_release_mock = mocker.patch(
        FROM_RELEASE_PATCH_PATH,
        return_value=expected_classifier,
    )

    result = construct_classifier(
        classifier_release_directory=release_directory,
        classifier_device="cpu",
    )

    assert result is expected_classifier

    from_release_mock.assert_called_once_with(
        release_directory=release_directory,
        device="cpu",
    )


def test_construct_classifier_passes_automatic_device_selection(
    mocker,
) -> None:
    expected_classifier = mocker.Mock()

    from_release_mock = mocker.patch(
        FROM_RELEASE_PATCH_PATH,
        return_value=expected_classifier,
    )

    result = construct_classifier(
        classifier_release_directory=(
            "/weights/classifier_release"
        ),
        classifier_device="auto",
    )

    assert result is expected_classifier

    from_release_mock.assert_called_once_with(
        release_directory=(
            "/weights/classifier_release"
        ),
        device="auto",
    )


@pytest.mark.parametrize(
    "loading_error",
    [
        FileNotFoundError("Release directory does not exist"),
        ValueError("Invalid classifier metadata"),
        RuntimeError("Unable to load model weights"),
    ],
)
def test_construct_classifier_translates_loading_failure(
    mocker,
    loading_error: Exception,
) -> None:
    from_release_mock = mocker.patch(
        FROM_RELEASE_PATCH_PATH,
        side_effect=loading_error,
    )

    with pytest.raises(
        ClassifierLoadingError,
        match="Classifier model failed to load",
    ) as exception_info:
        construct_classifier(
            classifier_release_directory=(
                "/weights/classifier_release"
            ),
            classifier_device="cpu",
        )

    assert exception_info.value.__cause__ is loading_error

    from_release_mock.assert_called_once_with(
        release_directory=(
            "/weights/classifier_release"
        ),
        device="cpu",
    )


def test_predict_calls_released_classifier_with_input_array(
    mocker,
) -> None:
    input_array = np.zeros(
        (210, 68),
        dtype=np.float32,
    )

    model_result = SimpleNamespace(
        logit=1.25,
        probability=0.7772998611746911,
        prediction=1,
        class_name="attempt",
        threshold=0.55,
    )

    classifier = mocker.Mock()
    classifier.predict.return_value = model_result

    result = predict(
        classifier=classifier,
        input_array=input_array,
    )

    classifier.predict.assert_called_once_with(
        model_input=input_array,
    )

    passed_input_array = (
        classifier.predict.call_args.kwargs[
            "model_input"
        ]
    )

    assert passed_input_array is input_array
    assert isinstance(
        result,
        ClipClassificationResult,
    )
    assert result.logit == 1.25
    assert result.probability == pytest.approx(
        0.7772998611746911
    )
    assert result.prediction == 1
    assert result.class_name == "attempt"
    assert result.threshold == pytest.approx(0.55)


@pytest.mark.parametrize(
    (
        "model_result",
        "expected_values",
    ),
    [
        (
            SimpleNamespace(
                logit=-1.5,
                probability=0.18242552380635635,
                prediction=0,
                class_name="no_attempt",
                threshold=0.55,
            ),
            {
                "logit": -1.5,
                "probability": 0.18242552380635635,
                "prediction": 0,
                "class_name": "no_attempt",
                "threshold": 0.55,
            },
        ),
        (
            SimpleNamespace(
                logit=0.75,
                probability=0.679178699175393,
                prediction=1,
                class_name="attempt",
                threshold=0.55,
            ),
            {
                "logit": 0.75,
                "probability": 0.679178699175393,
                "prediction": 1,
                "class_name": "attempt",
                "threshold": 0.55,
            },
        ),
    ],
)
def test_predict_translates_package_result_to_internal_result(
    mocker,
    model_result: SimpleNamespace,
    expected_values: dict[str, object],
) -> None:
    classifier = mocker.Mock()
    classifier.predict.return_value = model_result

    input_array = np.ones(
        (210, 68),
        dtype=np.float32,
    )

    result = predict(
        classifier=classifier,
        input_array=input_array,
    )

    assert isinstance(
        result,
        ClipClassificationResult,
    )
    assert result.logit == expected_values["logit"]
    assert result.probability == pytest.approx(
        expected_values["probability"]
    )
    assert (
        result.prediction
        == expected_values["prediction"]
    )
    assert (
        result.class_name
        == expected_values["class_name"]
    )
    assert result.threshold == pytest.approx(
        expected_values["threshold"]
    )


@pytest.mark.parametrize(
    "classifier_error",
    [
        TypeError("model_input must be a NumPy array"),
        ValueError(
            "Expected input shape (210, 68)"
        ),
    ],
)
def test_predict_translates_invalid_input_failure(
    mocker,
    classifier_error: Exception,
) -> None:
    classifier = mocker.Mock()
    classifier.predict.side_effect = (
        classifier_error
    )

    input_array = np.zeros(
        (209, 68),
        dtype=np.float32,
    )

    with pytest.raises(
        InvalidClassifierInputError,
        match=(
            "Classifier input must be a NumPy array "
            r"with shape \(210, 68\)"
        ),
    ) as exception_info:
        predict(
            classifier=classifier,
            input_array=input_array,
        )

    assert (
        exception_info.value.__cause__
        is classifier_error
    )

    classifier.predict.assert_called_once_with(
        model_input=input_array,
    )


def test_predict_does_not_translate_unexpected_classifier_failure(
    mocker,
) -> None:
    classifier_error = RuntimeError(
        "Unexpected inference failure"
    )

    classifier = mocker.Mock()
    classifier.predict.side_effect = (
        classifier_error
    )

    input_array = np.zeros(
        (210, 68),
        dtype=np.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected inference failure",
    ) as exception_info:
        predict(
            classifier=classifier,
            input_array=input_array,
        )

    assert exception_info.value is classifier_error
