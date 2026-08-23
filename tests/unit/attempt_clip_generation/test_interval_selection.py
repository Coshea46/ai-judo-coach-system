import pytest

from ai_judo_coach.attempt_clip_generation.interval_selection import (
    select_new_intervals,
)
from ai_judo_coach.exceptions import (
    NoSurvivingWindowsError,
)
from ai_judo_coach.schemas.internal import (
    DetectedAttemptWindow,
    InitialClipWindow,
    SelectedInterval,
)


def _create_detected_window(
    window_id: int,
    start_time: float,
    end_time: float,
    attempt_probability: float,
) -> DetectedAttemptWindow:
    """Create one detected attempt window for testing."""

    return DetectedAttemptWindow(
        window=InitialClipWindow(
            start_time=start_time,
            end_time=end_time,
            window_id=window_id,
        ),
        attempt_probability=attempt_probability,
    )


def test_select_new_intervals_sorts_and_merges_touching_windows() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=2,
            start_time=20.0,
            end_time=27.0,
            attempt_probability=0.80,
        ),
        _create_detected_window(
            window_id=1,
            start_time=7.0,
            end_time=14.0,
            attempt_probability=0.70,
        ),
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.60,
        ),
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=30.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=10,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=0.0,
            end_time_seconds=14.0,
        ),
        SelectedInterval(
            clip_id="1",
            start_time_seconds=20.0,
            end_time_seconds=27.0,
        ),
    ]


def test_select_new_intervals_merges_overlapping_windows() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.60,
        ),
        _create_detected_window(
            window_id=1,
            start_time=3.0,
            end_time=10.0,
            attempt_probability=0.90,
        ),
        _create_detected_window(
            window_id=2,
            start_time=6.0,
            end_time=13.0,
            attempt_probability=0.70,
        ),
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=20.0,
        max_new_interval_duration=20.0,
        max_intervals_per_video=10,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=0.0,
            end_time_seconds=13.0,
        ),
    ]


@pytest.mark.parametrize(
    (
        "peak_window_index",
        "expected_start",
        "expected_end",
    ),
    [
        (0, 0.0, 14.0),
        (2, 2.5, 16.5),
        (4, 5.0, 19.0),
    ],
)
def test_select_new_intervals_caps_merged_window_around_peak(
    peak_window_index: int,
    expected_start: float,
    expected_end: float,
) -> None:
    window_bounds = [
        (0.0, 7.0),
        (3.0, 10.0),
        (6.0, 13.0),
        (9.0, 16.0),
        (12.0, 19.0),
    ]

    surviving_windows = [
        _create_detected_window(
            window_id=window_index,
            start_time=start_time,
            end_time=end_time,
            attempt_probability=(
                0.90
                if window_index == peak_window_index
                else 0.10
            ),
        )
        for window_index, (start_time, end_time)
        in enumerate(window_bounds)
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=30.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=10,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=expected_start,
            end_time_seconds=expected_end,
        ),
    ]


def test_select_new_intervals_clamps_interval_to_video_duration() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=25.0,
            end_time=35.0,
            attempt_probability=0.80,
        ),
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=30.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=10,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=25.0,
            end_time_seconds=30.0,
        ),
    ]


def test_select_new_intervals_retains_strongest_intervals() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.40,
        ),
        _create_detected_window(
            window_id=1,
            start_time=10.0,
            end_time=17.0,
            attempt_probability=0.90,
        ),
        _create_detected_window(
            window_id=2,
            start_time=20.0,
            end_time=27.0,
            attempt_probability=0.70,
        ),
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=30.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=2,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=10.0,
            end_time_seconds=17.0,
        ),
        SelectedInterval(
            clip_id="1",
            start_time_seconds=20.0,
            end_time_seconds=27.0,
        ),
    ]


def test_select_new_intervals_breaks_probability_ties_chronologically() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=20.0,
            end_time=27.0,
            attempt_probability=0.80,
        ),
        _create_detected_window(
            window_id=1,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.80,
        ),
        _create_detected_window(
            window_id=2,
            start_time=10.0,
            end_time=17.0,
            attempt_probability=0.80,
        ),
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=30.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=2,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=0.0,
            end_time_seconds=7.0,
        ),
        SelectedInterval(
            clip_id="1",
            start_time_seconds=10.0,
            end_time_seconds=17.0,
        ),
    ]


def test_select_new_intervals_accepts_probability_boundaries() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.0,
        ),
        _create_detected_window(
            window_id=1,
            start_time=10.0,
            end_time=17.0,
            attempt_probability=1.0,
        ),
    ]

    result = select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=20.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=10,
    )

    assert result == [
        SelectedInterval(
            clip_id="0",
            start_time_seconds=0.0,
            end_time_seconds=7.0,
        ),
        SelectedInterval(
            clip_id="1",
            start_time_seconds=10.0,
            end_time_seconds=17.0,
        ),
    ]


def test_select_new_intervals_does_not_modify_input_order() -> None:
    first_window = _create_detected_window(
        window_id=1,
        start_time=10.0,
        end_time=17.0,
        attempt_probability=0.80,
    )
    second_window = _create_detected_window(
        window_id=0,
        start_time=0.0,
        end_time=7.0,
        attempt_probability=0.70,
    )

    surviving_windows = [
        first_window,
        second_window,
    ]

    select_new_intervals(
        surviving_initial_windows=surviving_windows,
        source_video_duration=20.0,
        max_new_interval_duration=14.0,
        max_intervals_per_video=10,
    )

    assert surviving_windows == [
        first_window,
        second_window,
    ]


def test_select_new_intervals_rejects_empty_window_list() -> None:
    with pytest.raises(
        NoSurvivingWindowsError,
        match="surviving initial windows array is empty",
    ):
        select_new_intervals(
            surviving_initial_windows=[],
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )


@pytest.mark.parametrize(
    "source_video_duration",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_select_new_intervals_rejects_invalid_video_duration(
    source_video_duration: float,
) -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "source_video_duration must be a finite number "
            "greater than zero"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=source_video_duration,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )


@pytest.mark.parametrize(
    "max_new_interval_duration",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_select_new_intervals_rejects_invalid_maximum_duration(
    max_new_interval_duration: float,
) -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "max_new_interval_duration must be a finite number "
            "greater than zero"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=max_new_interval_duration,
            max_intervals_per_video=10,
        )


@pytest.mark.parametrize(
    "max_intervals_per_video",
    [
        0,
        -1,
        1.5,
        True,
        "2",
        None,
    ],
)
def test_select_new_intervals_rejects_invalid_interval_limit(
    max_intervals_per_video: object,
) -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "max_intervals_per_video must be an integer "
            "greater than zero"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=max_intervals_per_video,
        )


@pytest.mark.parametrize(
    (
        "start_time",
        "end_time",
    ),
    [
        (float("nan"), 7.0),
        (float("inf"), 7.0),
        (float("-inf"), 7.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, float("-inf")),
    ],
)
def test_select_new_intervals_rejects_non_finite_window_times(
    start_time: float,
    end_time: float,
) -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=start_time,
            end_time=end_time,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Detected window start and end times must be finite"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )


def test_select_new_intervals_rejects_negative_window_start() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=-1.0,
            end_time=6.0,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Detected window start time must be zero or greater"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )


@pytest.mark.parametrize(
    (
        "start_time",
        "end_time",
    ),
    [
        (4.0, 4.0),
        (5.0, 4.0),
    ],
)
def test_select_new_intervals_rejects_non_positive_window_duration(
    start_time: float,
    end_time: float,
) -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=start_time,
            end_time=end_time,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Detected window end time must be greater than "
            "its start time"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )


@pytest.mark.parametrize(
    "attempt_probability",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_select_new_intervals_rejects_invalid_probability(
    attempt_probability: float,
) -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=0.0,
            end_time=7.0,
            attempt_probability=attempt_probability,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Detected window attempt probability must be "
            "between 0.0 and 1.0"
        ),
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )


def test_select_new_intervals_rejects_window_outside_video() -> None:
    surviving_windows = [
        _create_detected_window(
            window_id=0,
            start_time=30.0,
            end_time=37.0,
            attempt_probability=0.80,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Merged interval is empty after clamping",
    ):
        select_new_intervals(
            surviving_initial_windows=surviving_windows,
            source_video_duration=30.0,
            max_new_interval_duration=14.0,
            max_intervals_per_video=10,
        )
