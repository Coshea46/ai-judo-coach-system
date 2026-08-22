"""Schemas in this file are for storing the output of the player detection package"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PlayerPoseSequence:
    """
    Stores one player's sequence of poses across a clip
    """

    keypoints_xy_px: np.ndarray       # [T, 17, 2], float32
    keypoints_xy_norm: np.ndarray     # [T, 17, 2], float32
    keypoints_conf: np.ndarray        # [T, 17], float32

    # mask for if player detection missing in frame
    # true means player not detected in frame
    missing_mask: np.ndarray          # [T], bool

    # stores index of pose in FrameDetections person_detections array that each pose was from
    source_detection_idx: np.ndarray  # [T], int32

    # bytetrack id of each pose
    source_track_id: np.ndarray       # [T], int32



@dataclass(slots=True)
class TwoPlayerPoseSequences:
    """
    Stores the PlayerPoseSequence objects for both players
    in a given clip
    """

    clip_id: str
    player_a_pose_sequence: PlayerPoseSequence
    player_b_pose_sequence: PlayerPoseSequence

