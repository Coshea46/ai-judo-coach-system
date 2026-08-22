import torch


def resolve_yolo_device(
    requested_device: str,
) -> str:
    """Resolve the configured device for YOLO inference."""

    normalised_device = requested_device.strip().lower()

    if normalised_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    return normalised_device
