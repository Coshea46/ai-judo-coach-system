import pytest

from ai_judo_coach.inference.yolo_feeder.device import (
    resolve_yolo_device,
)


CUDA_AVAILABLE_PATCH_PATH = (
    "ai_judo_coach.inference.yolo_feeder."
    "device.torch.cuda.is_available"
)


@pytest.mark.parametrize(
    (
        "cuda_available",
        "expected_device",
    ),
    [
        (
            True,
            "cuda:0",
        ),
        (
            False,
            "cpu",
        ),
    ],
)
def test_resolve_yolo_device_resolves_auto_device(
    mocker,
    cuda_available: bool,
    expected_device: str,
) -> None:
    cuda_available_mock = mocker.patch(
        CUDA_AVAILABLE_PATCH_PATH,
        return_value=cuda_available,
    )

    result = resolve_yolo_device(
        requested_device="auto",
    )

    assert result == expected_device
    cuda_available_mock.assert_called_once_with()


@pytest.mark.parametrize(
    "requested_device",
    [
        "auto",
        "AUTO",
        " Auto ",
        "\taUtO\n",
    ],
)
def test_resolve_yolo_device_normalises_auto_value(
    mocker,
    requested_device: str,
) -> None:
    cuda_available_mock = mocker.patch(
        CUDA_AVAILABLE_PATCH_PATH,
        return_value=True,
    )

    result = resolve_yolo_device(
        requested_device=requested_device,
    )

    assert result == "cuda:0"
    cuda_available_mock.assert_called_once_with()


@pytest.mark.parametrize(
    (
        "requested_device",
        "expected_device",
    ),
    [
        (
            "cpu",
            "cpu",
        ),
        (
            " CPU ",
            "cpu",
        ),
        (
            "cuda",
            "cuda",
        ),
        (
            "CUDA:0",
            "cuda:0",
        ),
        (
            " Cuda:1 ",
            "cuda:1",
        ),
        (
            "mps",
            "mps",
        ),
    ],
)
def test_resolve_yolo_device_returns_normalised_explicit_device(
    mocker,
    requested_device: str,
    expected_device: str,
) -> None:
    cuda_available_mock = mocker.patch(
        CUDA_AVAILABLE_PATCH_PATH,
    )

    result = resolve_yolo_device(
        requested_device=requested_device,
    )

    assert result == expected_device
    cuda_available_mock.assert_not_called()
