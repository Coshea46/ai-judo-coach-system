from operator import attrgetter
from dataclasses import dataclass
import math

from ai_judo_coach.schemas.internal import (
    DetectedAttemptWindow,
    SelectedInterval,
)
from ai_judo_coach.exceptions import (
    NoSurvivingWindowsError,
)


@dataclass(slots=True, frozen=True)
class _MergedWindow:
    """
    Local dataclass for 
    intermediate storage of
    merged initial windows
    """

    window_start_sec: float
    window_end_sec: float
    peak_window_probability: float
    peak_window_start_sec: float
    peak_window_end_sec: float


def select_new_intervals(
    surviving_initial_windows: list[DetectedAttemptWindow],
    source_video_duration: float,
    max_new_interval_duration: float,
    max_intervals_per_video: int,
) -> list[SelectedInterval]:
    """
    Select final attempt intervals from surviving initial windows.

    Overlapping or touching windows are merged. Merged intervals that
    exceed the maximum duration are capped around their highest-scoring
    contributing window. If too many intervals remain, those with the
    highest peak probabilities are retained and returned in chronological
    order.
    """

    num_surviving_initial_windows = len(
        surviving_initial_windows
    )

    # Defensive check.
    if num_surviving_initial_windows < 1:
        raise NoSurvivingWindowsError(
            "surviving initial windows array is empty"
        )

    _validate_selection_arguments(
        source_video_duration=source_video_duration,
        max_new_interval_duration=max_new_interval_duration,
        max_intervals_per_video=max_intervals_per_video,
    )

    # Sort so only one sweep is needed when merging intervals.
    get_start = attrgetter("window.start_time")
    get_end = attrgetter("window.end_time")

    sorted_initial_windows = sorted(
        surviving_initial_windows,
        key=lambda window: (
            get_start(window),
            get_end(window),
        ),
    )

    _validate_detected_windows(
        detected_windows=sorted_initial_windows,
    )

    merged_windows = _merge_detected_windows(
        sorted_initial_windows=sorted_initial_windows,
    )

    capped_windows = [
        _cap_merged_window(
            merged_window=merged_window,
            source_video_duration=source_video_duration,
            max_new_interval_duration=(
                max_new_interval_duration
            ),
        )
        for merged_window in merged_windows
    ]

    # Retain the strongest intervals if more were found than the
    # configured maximum.
    strongest_windows = sorted(
        capped_windows,
        key=lambda window: (
            -window.peak_window_probability,
            window.window_start_sec,
            window.window_end_sec,
        ),
    )[:max_intervals_per_video]

    # Return the final intervals in chronological order.
    selected_windows = sorted(
        strongest_windows,
        key=lambda window: (
            window.window_start_sec,
            window.window_end_sec,
        ),
    )

    return [
        SelectedInterval(
            clip_id=str(interval_index),
            start_time_seconds=(
                selected_window.window_start_sec
            ),
            end_time_seconds=(
                selected_window.window_end_sec
            ),
        )
        for interval_index, selected_window
        in enumerate(selected_windows)
    ]


def _merge_detected_windows(
    sorted_initial_windows: list[DetectedAttemptWindow],
) -> list[_MergedWindow]:
    """Merge overlapping or touching positively classified windows."""

    first_window = sorted_initial_windows[0]

    current_start = float(
        first_window.window.start_time
    )
    current_end = float(
        first_window.window.end_time
    )
    current_peak_probability = float(
        first_window.attempt_probability
    )
    current_peak_start = current_start
    current_peak_end = current_end

    merged_windows: list[_MergedWindow] = []

    for next_window in sorted_initial_windows[1:]:
        next_start = float(
            next_window.window.start_time
        )
        next_end = float(
            next_window.window.end_time
        )
        next_probability = float(
            next_window.attempt_probability
        )

        if next_start <= current_end:
            # The next window overlaps or touches the current merged
            # region, so extend the region if necessary.
            current_end = max(
                current_end,
                next_end,
            )

            # Retain the strongest contributing source window.
            if next_probability > current_peak_probability:
                current_peak_probability = next_probability
                current_peak_start = next_start
                current_peak_end = next_end

            continue

        # The next window does not overlap the current region, so the
        # current merged region is complete.
        merged_windows.append(
            _MergedWindow(
                window_start_sec=current_start,
                window_end_sec=current_end,
                peak_window_probability=(
                    current_peak_probability
                ),
                peak_window_start_sec=current_peak_start,
                peak_window_end_sec=current_peak_end,
            )
        )

        # Begin a new merged region from the next window.
        current_start = next_start
        current_end = next_end
        current_peak_probability = next_probability
        current_peak_start = next_start
        current_peak_end = next_end

    # The final region is not followed by another window that would
    # cause it to be appended inside the loop.
    merged_windows.append(
        _MergedWindow(
            window_start_sec=current_start,
            window_end_sec=current_end,
            peak_window_probability=(
                current_peak_probability
            ),
            peak_window_start_sec=current_peak_start,
            peak_window_end_sec=current_peak_end,
        )
    )

    return merged_windows


def _cap_merged_window(
    merged_window: _MergedWindow,
    source_video_duration: float,
    max_new_interval_duration: float,
) -> _MergedWindow:
    """
    Clamp one merged interval to the source video and maximum duration.

    Intervals longer than the configured maximum are centred around
    their highest-scoring contributing window.
    """

    clamped_start = max(
        0.0,
        merged_window.window_start_sec,
    )
    clamped_end = min(
        source_video_duration,
        merged_window.window_end_sec,
    )

    if clamped_end <= clamped_start:
        raise ValueError(
            "Merged interval is empty after clamping to the "
            "source video: "
            f"start={clamped_start}, end={clamped_end}"
        )

    clamped_duration = clamped_end - clamped_start

    if clamped_duration <= max_new_interval_duration:
        return _MergedWindow(
            window_start_sec=clamped_start,
            window_end_sec=clamped_end,
            peak_window_probability=(
                merged_window.peak_window_probability
            ),
            peak_window_start_sec=(
                merged_window.peak_window_start_sec
            ),
            peak_window_end_sec=(
                merged_window.peak_window_end_sec
            ),
        )

    peak_window_centre = (
        merged_window.peak_window_start_sec
        + merged_window.peak_window_end_sec
    ) / 2.0

    proposed_start = (
        peak_window_centre
        - (max_new_interval_duration / 2.0)
    )

    minimum_start = clamped_start
    maximum_start = (
        clamped_end
        - max_new_interval_duration
    )

    selected_start = min(
        max(proposed_start, minimum_start),
        maximum_start,
    )

    selected_end = (
        selected_start
        + max_new_interval_duration
    )

    return _MergedWindow(
        window_start_sec=selected_start,
        window_end_sec=selected_end,
        peak_window_probability=(
            merged_window.peak_window_probability
        ),
        peak_window_start_sec=(
            merged_window.peak_window_start_sec
        ),
        peak_window_end_sec=(
            merged_window.peak_window_end_sec
        ),
    )


def _validate_selection_arguments(
    source_video_duration: float,
    max_new_interval_duration: float,
    max_intervals_per_video: int,
) -> None:
    """Validate interval-selection configuration values."""

    if (
        not math.isfinite(source_video_duration)
        or source_video_duration <= 0.0
    ):
        raise ValueError(
            "source_video_duration must be a finite number "
            "greater than zero"
        )

    if (
        not math.isfinite(max_new_interval_duration)
        or max_new_interval_duration <= 0.0
    ):
        raise ValueError(
            "max_new_interval_duration must be a finite number "
            "greater than zero"
        )

    if (
        isinstance(max_intervals_per_video, bool)
        or not isinstance(max_intervals_per_video, int)
        or max_intervals_per_video <= 0
    ):
        raise ValueError(
            "max_intervals_per_video must be an integer "
            "greater than zero"
        )


def _validate_detected_windows(
    detected_windows: list[DetectedAttemptWindow],
) -> None:
    """Validate the temporal bounds and probability of each window."""

    for detected_window in detected_windows:
        window_start = float(
            detected_window.window.start_time
        )
        window_end = float(
            detected_window.window.end_time
        )
        attempt_probability = float(
            detected_window.attempt_probability
        )

        if (
            not math.isfinite(window_start)
            or not math.isfinite(window_end)
        ):
            raise ValueError(
                "Detected window start and end times must be finite"
            )

        if window_start < 0.0:
            raise ValueError(
                "Detected window start time must be zero or greater"
            )

        if window_end <= window_start:
            raise ValueError(
                "Detected window end time must be greater than "
                "its start time"
            )

        if (
            not math.isfinite(attempt_probability)
            or not 0.0 <= attempt_probability <= 1.0
        ):
            raise ValueError(
                "Detected window attempt probability must be "
                "between 0.0 and 1.0"
            )
