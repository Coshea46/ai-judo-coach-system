from collections.abc import Iterator

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


def track_video(
    yolo_model: YOLO,
    tracker_path: str,
    clip_as_numpy: list[np.ndarray],
    compute_device: str | int,
) -> Iterator[Results]:
    """
    Run YOLO tracking on a video and return a single-use Results iterator.

    This function intentionally returns the raw Ultralytics stream.
    Conversion into project schemas happens in results_adapter.py.
    """

    results_stream = yolo_model.track(
        source=clip_as_numpy,
        stream=True,
        tracker=tracker_path,
        device=compute_device,
        persist=False
    )

    return results_stream
