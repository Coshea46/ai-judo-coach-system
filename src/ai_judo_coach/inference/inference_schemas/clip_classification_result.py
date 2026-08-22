"""Application-level clip-classification result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipClassificationResult:
    """Classification result exposed to the wider inference package."""

    logit: float
    probability: float
    prediction: int
    class_name: str
    threshold: float
