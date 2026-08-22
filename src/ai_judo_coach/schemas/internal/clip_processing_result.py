from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipProcessingResult:
    """Public result of processing one complete clip."""

    clip_id: str
    contains_throw_attempt: bool
    attempt_probability: float
    predicted_class_name: str
