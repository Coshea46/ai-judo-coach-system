import numpy as np

from ai_judo_coach.inference.inference_schemas import (
    PlayerPoseSequence,
    TwoPlayerPoseSequences,
    ClipDetections,
    FrameDetections,
    PersonDetection
)
from .candidate_states import (
    CandidateState,
    is_assignment_missing,
)
from .tracking_config import PlayerDetectionConfig


def build_two_player_pose_sequences(
    clip_detections: ClipDetections,
    frame_player_state_sequence: list[CandidateState],
    config: PlayerDetectionConfig
) -> TwoPlayerPoseSequences:
    """
    Converts a list of states for each
    frame into the pose sequences
    that those states represent.

    Expects clip_detections.frame_detections
    list and frame_player_state_sequence list
    to be index aligned.
    """

    if (
        len(frame_player_state_sequence)
        != len(clip_detections.frame_detections)
    ):
        raise ValueError(
            "frame_player_state_sequence must contain exactly "
            "one state for each frame in clip_detections."
        )

    player_a_detection_sequence: list[PersonDetection] = []
    player_b_detection_sequence: list[PersonDetection] = []

    for frame_idx, state in enumerate(frame_player_state_sequence):

        player_a = _resolve_player_in_frame(
            frame_detections=clip_detections.frame_detections[frame_idx],
            index_in_person_detection_list=state[0],
            config=config
        )

        player_b = _resolve_player_in_frame(
            frame_detections=clip_detections.frame_detections[frame_idx],
            index_in_person_detection_list=state[1],
            config=config
        )

        player_a_detection_sequence.append(player_a)
        player_b_detection_sequence.append(player_b)

    # now turn PersonDetection sequences into PlayerPoseSequence and TwoPlayerPoseSequences objects
    player_a_pose_sequence: PlayerPoseSequence = (
        _person_detection_sequence_to_player_pose_sequence(
            player_detection_sequence=player_a_detection_sequence,
            missing_detection_sentinel=config.missing_detection_sentinel
        )
    )

    player_b_pose_sequence: PlayerPoseSequence = (
        _person_detection_sequence_to_player_pose_sequence(
            player_detection_sequence=player_b_detection_sequence,
            missing_detection_sentinel=config.missing_detection_sentinel
        )
    )

    clip_two_player_sequence: TwoPlayerPoseSequences = TwoPlayerPoseSequences(
        clip_id=clip_detections.clip_id,
        player_a_pose_sequence=player_a_pose_sequence,
        player_b_pose_sequence=player_b_pose_sequence
    )

    return clip_two_player_sequence


# needs config as needs to know what the missing_detection_sentinel value is
def _resolve_player_in_frame(
    frame_detections: FrameDetections,
    index_in_person_detection_list: int,
    config: PlayerDetectionConfig
) -> PersonDetection:
    """
    Resolves the person detection object
    for a single player in a frame,
    given a player index within a state.

    If a player has no assigned pose for 
    that frame, its PersonDetection numpy arrays
    (e.g. keypoint arrays) are assigned as
    numpy arrays containing nan values to pad
    array length. Its detection index and track id
    will also take on the value of the missing
    detection sentinel
    """

    if is_assignment_missing(
        assignment_idx=index_in_person_detection_list,
        config=config
    ):
        # passing same nan numpy array object here should be fine as only dataclass constructor
        bbox_nan_numpy_array = np.full(
            (4,),
            np.nan,
            dtype=np.float32
        )

        coco_kp_nan_numpy_array = np.full(
            (17, 2),
            np.nan,
            dtype=np.float32
        )

        coco_confidence_zeros_numpy_array = np.zeros(
            (17,),
            dtype=np.float32
        )

        return PersonDetection(
            detection_idx=config.missing_detection_sentinel,
            track_id=config.missing_detection_sentinel,
            bbox_xyxy_px=bbox_nan_numpy_array,
            bbox_xyxy_normalized=bbox_nan_numpy_array,
            bbox_conf=0.0,
            keypoints_xy_px=coco_kp_nan_numpy_array,
            keypoints_xy_norm=coco_kp_nan_numpy_array,
            keypoints_conf=coco_confidence_zeros_numpy_array
        )

    if not (
        0
        <= index_in_person_detection_list
        < len(frame_detections.person_detections)
    ):
        raise IndexError(
            "Player assignment index is outside the "
            "FrameDetections.person_detections list."
        )

    return frame_detections.person_detections[
        index_in_person_detection_list
    ]


def _person_detection_sequence_to_player_pose_sequence(
    player_detection_sequence: list[PersonDetection],
    missing_detection_sentinel: int
) -> PlayerPoseSequence:
    """
    Converts a single player's sequence of PersonDetection
    objects into a PlayerPoseSequence object
    """

    all_keypoints_xy_px: list[np.ndarray] = []
    all_keypoints_xy_norm: list[np.ndarray] = []
    all_keypoints_conf: list[np.ndarray] = []
    all_source_detection_idxs: list[int] = []
    all_source_track_ids: list[int] = []

    missing_mask: list[bool] = []

    coco_confidence_zeros_numpy_array = np.zeros(
        (17,),
        dtype=np.float32
    )

    for player_in_frame in player_detection_sequence:

        all_keypoints_xy_px.append(player_in_frame.keypoints_xy_px)
        all_keypoints_xy_norm.append(player_in_frame.keypoints_xy_norm)
        all_keypoints_conf.append(player_in_frame.keypoints_conf)
        all_source_detection_idxs.append(player_in_frame.detection_idx)

        source_track_id = player_in_frame.track_id

        if source_track_id is None:
            source_track_id = missing_detection_sentinel

        all_source_track_ids.append(source_track_id)

        if np.array_equal(
            player_in_frame.keypoints_conf,
            coco_confidence_zeros_numpy_array
        ):
            missing_mask.append(True)

        else:
            missing_mask.append(False)

    return PlayerPoseSequence(
        keypoints_xy_px=np.asarray(
            all_keypoints_xy_px,
            dtype=np.float32
        ),
        keypoints_xy_norm=np.asarray(
            all_keypoints_xy_norm,
            dtype=np.float32
        ),
        keypoints_conf=np.asarray(
            all_keypoints_conf,
            dtype=np.float32
        ),
        missing_mask=np.asarray(
            missing_mask,
            dtype=np.bool_
        ),
        source_detection_idx=np.asarray(
            all_source_detection_idxs,
            dtype=np.int32
        ),
        source_track_id=np.asarray(
            all_source_track_ids,
            dtype=np.int32
        )
    )
