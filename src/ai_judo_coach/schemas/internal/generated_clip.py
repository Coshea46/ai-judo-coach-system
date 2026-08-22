from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectedInterval:
    """
    One interval selected from the
    surviving intial windows
    """

    clip_id: str
    start_time_seconds: float
    end_time_seconds: float



@dataclass(frozen=True, slots=True)
class GeneratedAttemptClip:
    """One final attempt clip generated in backend storage."""

    clip_id: str
    start_time_seconds: float   # start time in source video
    end_time_seconds: float     # end time in source video
    file_path: str
