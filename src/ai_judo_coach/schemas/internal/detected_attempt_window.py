from dataclasses import dataclass
from .initial_clip_window import InitialClipWindow


@dataclass(frozen=True, slots=True)
class DetectedAttemptWindow:
    """One initial window classified as containing an attempt."""

    window: InitialClipWindow
    attempt_probability: float
