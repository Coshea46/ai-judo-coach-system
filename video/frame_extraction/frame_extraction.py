import numpy as np

# NEED TO DECIDE IF USING DECORD OR OPENCV HERE
# CURRENTLY LEANING TOWARDS DECORD BUT GOING TO DO A QUICK BENCHMARK FIRST

from schemas.internal import InitialClipWindow

# fps should be passed in by caller and caller should be the one to get from config.py
def extract_frames_from_initial_window(source_video_path: str, window: InitialClipWindow, video_fps: int) -> list[np.ndarray]:
    """
    Function for extracting the frames within a single
    initial clip window and storing as a list of numpy arrays

    Each frame gets its own numpy array, giving a list of numpy arrays.
    Ultralytics YOLO v11 expects a list of numpy arrays as input.
    """


