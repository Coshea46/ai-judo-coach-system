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

    Frames are supplied individually to avoid Ultralytics treating the
    complete list as one inference batch. Tracker state is retained between
    frames in the clip and reset before processing a new clip.

    This function intentionally returns the raw Ultralytics stream.
    Conversion into project schemas happens in results_adapter.py.
    """

    _reset_existing_trackers(
        yolo_model=yolo_model,
    )

    return _track_frames(
        yolo_model=yolo_model,
        tracker_path=tracker_path,
        clip_as_numpy=clip_as_numpy,
        compute_device=compute_device,
    )


def _track_frames(
    yolo_model: YOLO,
    tracker_path: str,
    clip_as_numpy: list[np.ndarray],
    compute_device: str | int,
) -> Iterator[Results]:
    """
    Track the clip one frame at a time.

    persist=True retains ByteTrack state between these individual
    inference calls. State was explicitly reset before this iterator
    was constructed.
    """

    for frame in clip_as_numpy:
        frame_results = yolo_model.track(
            source=frame,
            stream=True,
            tracker=tracker_path,
            device=compute_device,
            persist=True,
        )

        yield from frame_results


def _reset_existing_trackers(
    yolo_model: YOLO,
) -> None:
    """Reset tracker state left by a previously processed clip."""

    predictor = getattr(
        yolo_model,
        "predictor",
        None,
    )

    if predictor is None:
        return

    trackers = getattr(
        predictor,
        "trackers",
        None,
    )

    if trackers is None:
        return

    for tracker in trackers:
        tracker.reset()
