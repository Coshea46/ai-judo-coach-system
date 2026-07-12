from collections.abc import Iterator

import ffmpeg

from schemas.internal import InitialClipWindow
from exceptions.video import InvalidVideoError


def compute_initial_clip_windows(
    input_video_path: str, individual_window_duration: float, stride: float
) -> Iterator[InitialClipWindow]:
    """
    Splits a single video into overlapping time windows for downstream processing.

    Yields InitialClipWindow objects covering the full video duration,
    including a final window (which may overlap the previous one) to
    cover any remaining tail shorter than a full stride.
    """
    # output of this function should be a generator containing objects with type InitialClipWindow

    try:
        probe_details = ffmpeg.probe(input_video_path)

    except ffmpeg.Error as e:
        raise InvalidVideoError(
            f"ffmpeg could not read input video metadata: {e}"
        ) from e

    input_video_duration = float(probe_details["format"]["duration"])

    if input_video_duration < individual_window_duration:
        raise InvalidVideoError(
            f"Video duration ({input_video_duration}s) is shorter than "
            f"window duration ({individual_window_duration}s)"
        )

    # window_id is only unique within a single video's set of windows
    # keeping id's simple here so clips can be recombined easily later in pipeline
    video_playhead = 0.0  # tracks how far through the video the generator is
    window_idx = 0
    while video_playhead + individual_window_duration <= input_video_duration:

        clip_window = InitialClipWindow(
            start_time=video_playhead,
            end_time=video_playhead + individual_window_duration,
            window_id=window_idx,
        )

        yield clip_window

        video_playhead += stride
        window_idx += 1

    # create one last window if uncovered interval at end of input video
    if (input_video_duration - video_playhead) > 0:

        clip_window = InitialClipWindow(
            start_time=input_video_duration - individual_window_duration,
            end_time=input_video_duration,
            window_id=window_idx,
        )

        yield clip_window
