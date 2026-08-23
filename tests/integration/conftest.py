from pathlib import Path

import pytest


VIDEO_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "videos"
)



def _get_local_video_fixture(
    filename: str,
) -> Path:
    """
    Return the path to a local video fixture.

    Video fixtures are deliberately excluded from version control,
    so tests requiring an unavailable fixture are skipped.
    """

    video_path = (
        VIDEO_FIXTURE_DIRECTORY
        / filename
    )

    if not video_path.is_file():
        pytest.skip(
            "Local video fixture is unavailable: "
            f"{video_path}"
        )

    return video_path


@pytest.fixture
def attempt_video_path() -> Path:
    """Return the local clip containing a throw attempt."""

    return _get_local_video_fixture(
        "attempt_id24.mp4"
    )


@pytest.fixture
def no_throw_video_path() -> Path:
    """Return the local clip without a throw attempt."""

    return _get_local_video_fixture(
        "no_throw_clip_10.mp4"
    )


@pytest.fixture
def short_full_match_video_path() -> Path:
    """Return the local shorter full-match video."""

    return _get_local_video_fixture(
        "short_full_match_video.mp4"
    )


@pytest.fixture
def long_full_match_video_path() -> Path:
    """Return the local longer full-match video."""

    return _get_local_video_fixture(
        "long_full_match_video.mp4"
    )
