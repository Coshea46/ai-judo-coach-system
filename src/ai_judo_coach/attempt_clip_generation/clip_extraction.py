import ffmpeg

from ai_judo_coach.schemas.internal import(
    SelectedInterval,
    GeneratedAttemptClip
)


def extract_final_clips(
    selected_intervals: list[SelectedInterval]
) -> list[GeneratedAttemptClip]:
    """
    """
    