import numpy as np
import decord
from decord import VideoReader, cpu, gpu  # using decord since significantly faster and more accurate than OpenCV2
from decord._ffi.base import DECORDError

from ai_judo_coach.exceptions import InvalidFrameIndicesError
from ai_judo_coach.schemas.internal import InitialClipWindow


# fps should be passed in by caller and caller should be the one to get from config.py
def extract_frames_from_initial_window(
    source_video_path: str,
    window: InitialClipWindow,
    video_fps: float,
    device: str,
) -> list[np.ndarray]:
    """
    Function for extracting the frames within a single
    initial clip window and storing as a list of numpy arrays

    Each frame gets its own numpy array, giving a list of numpy arrays.
    Ultralytics YOLO v11 expects a list of numpy arrays as input.
    """

    video_reader = _read_video(
        source_video_path=source_video_path,
        desired_device=device,
    )

    start_frame = int(window.start_time * video_fps)
    end_frame = int(window.end_time * video_fps)

    frame_indices = list(
        range(
            start_frame,
            min(end_frame + 1, len(video_reader)),
        )
    )

    _check_frame_indices(
        desired_start_frame_idx=start_frame,
        desired_end_frame_idx=end_frame,
        found_frame_indices=frame_indices,
    )

    # decord returns rgb but ultralytics yolo v11 xl expects bgr format for frames
    bgr_frames = (
        video_reader.get_batch(indices=frame_indices)
        .asnumpy()[:, :, :, ::-1]
        .copy()
    )

    return [bgr_frames[i] for i in range(bgr_frames.shape[0])]


def _read_video(
    source_video_path: str,
    desired_device: str,
) -> VideoReader:
    """
    Reads the video at a defined
    filepath and returns as
    VideoReader instance
    """

    normalised_device = str(desired_device).lower().strip()
    gpu_requested = normalised_device.startswith(("gpu", "cuda"))

    device = _parse_desired_device(
        device_str=normalised_device,
    )

    try:
        return VideoReader(
            uri=source_video_path,
            ctx=device,
        )

    except DECORDError:
        if not gpu_requested:
            raise

        # default to cpu if gpu not available
        return VideoReader(
            uri=source_video_path,
            ctx=cpu(0),
        )


def _parse_desired_device(
    device_str: str,
) -> decord.ndarray.DECORDContext:
    """
    Parses the string literal representing
    the device to be used by decord
    (stored in the config.py file) and
    converts to the type expected by
    decord VideoReader constructor.

    Tries gpu first, defaults to cpu
    if gpu not available
    """

    device_str = str(device_str).lower().strip()

    if device_str.startswith(("gpu", "cuda")):
        device_id = (
            int(device_str.split(":")[-1])
            if ":" in device_str
            else 0
        )

        return gpu(device_id)

    return cpu(0)


def _check_frame_indices(
    desired_start_frame_idx: int,
    desired_end_frame_idx: int,
    found_frame_indices: list[int],
) -> None:
    """
    Defensive check for verifying
    that window frame indices
    are within the bounds of the
    frame indices found by decord
    """

    if desired_start_frame_idx < 0:
        raise InvalidFrameIndicesError(
            "Desired start frame index out of bounds"
        )

    if desired_end_frame_idx < 0:
        raise InvalidFrameIndicesError(
            "Desired end frame index out of bounds"
        )

    if desired_end_frame_idx < desired_start_frame_idx:
        raise InvalidFrameIndicesError(
            "Desired end frame index precedes start frame index"
        )

    if desired_start_frame_idx not in found_frame_indices:
        raise InvalidFrameIndicesError(
            "Desired start frame index out of bounds"
        )

    if desired_end_frame_idx not in found_frame_indices:
        raise InvalidFrameIndicesError(
            "Desired end frame index out of bounds"
        )
