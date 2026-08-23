import pytest

from ai_judo_coach.inference.yolo_feeder.model import (
    load_yolo_model,
)


YOLO_PATCH_PATH = (
    "ai_judo_coach.inference.yolo_feeder.model.YOLO"
)


def test_load_yolo_model_constructs_model_from_weights_path(
    mocker,
) -> None:
    yolo_model_path = (
        "weights/ultralytics_v11x_yolo/"
        "yolo11x-pose.pt"
    )
    expected_model = mocker.Mock()

    yolo_mock = mocker.patch(
        YOLO_PATCH_PATH,
        return_value=expected_model,
    )

    result = load_yolo_model(
        yolo_model_path=yolo_model_path,
    )

    assert result is expected_model
    yolo_mock.assert_called_once_with(
        yolo_model_path
    )


def test_load_yolo_model_propagates_loading_failure(
    mocker,
) -> None:
    loading_error = FileNotFoundError(
        "YOLO weights do not exist"
    )

    yolo_mock = mocker.patch(
        YOLO_PATCH_PATH,
        side_effect=loading_error,
    )

    yolo_model_path = "weights/missing.pt"

    with pytest.raises(
        FileNotFoundError,
        match="YOLO weights do not exist",
    ) as exception_info:
        load_yolo_model(
            yolo_model_path=yolo_model_path,
        )

    assert exception_info.value is loading_error
    yolo_mock.assert_called_once_with(
        yolo_model_path
    )
