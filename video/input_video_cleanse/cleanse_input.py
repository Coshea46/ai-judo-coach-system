from video.input_video_cleanse.normalize import normalize_video_fps
from video.input_video_cleanse.strip_audio import strip_audio
from config import TARGET_FPS


def cleanse_input_video(input_video_path: str) -> str:
    """
    Function for converting properties of input video from
    user into properties that the pipeline expects.

    Pipeline expects no audio and constant 30 fps.
    This creates new video with those exact properties,
    using the user's input video.
    """

    no_audio_video_path = strip_audio(input_video_path=input_video_path)

    normalized_no_audio_video_path = normalize_video_fps(
        input_video_path=no_audio_video_path, target_fps=TARGET_FPS
    )

    return normalized_no_audio_video_path
