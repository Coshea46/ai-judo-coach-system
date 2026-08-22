"""Takes a clip as list of numpy arrays all the way to its lstm classification"""

import numpy as np


def process_clip(
    clip_as_numpy: list[np.ndarray]
) -> bool:
    """
    
    """

    # MAYBE WANT TO RETURN A DATACLASS HERE FOR DIAGNOSTICS PURPOSES?

    # TODO: should be a mini pipeline taking a single clip to yolo to lstm input to lstm classification
    # the higher level orchestrator will handle the looping over clips