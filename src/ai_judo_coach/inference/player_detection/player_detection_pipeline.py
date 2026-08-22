from ai_judo_coach.inference.inference_schemas import(
    ClipDetections,
    TwoPlayerPoseSequences
)
from .tracking_config import PlayerDetectionConfig
from .candidate_states import generate_candidate_states_for_clip
from .solvers import viterbi_algorithm
from .pose_sequence_builder import build_two_player_pose_sequences
from .postprocess import(
    interpolate_two_player_pose_sequences_in_place,
    assess_pose_sequence_quality,
    PoseSequenceQualityReport
)


def detect_players(
    clip_detections: ClipDetections
) -> tuple[TwoPlayerPoseSequences, PoseSequenceQualityReport]:
    """
    Takes all of the poses found by yolo
    for a given clip and decides which 
    correspond to the 2 judo players in 
    each frame.

    Serves as entry point for player 
    detection system
    """

    # instantiate config class for player detection pipeline
    player_detection_config = PlayerDetectionConfig()

    candidate_states_for_clip = generate_candidate_states_for_clip(
        clip_detections=clip_detections,
        config=player_detection_config
    )

    most_likely_state_sequence = viterbi_algorithm(
        candidate_states_by_frame= candidate_states_for_clip,
        clip_detections=clip_detections,
        config=player_detection_config
    )

    clip_player_pose_sequence = build_two_player_pose_sequences(
        clip_detections=clip_detections,
        frame_player_state_sequence=most_likely_state_sequence,
        config=player_detection_config
    )

    interpolate_two_player_pose_sequences_in_place(
        clip_player_pose_sequences=clip_player_pose_sequence,
        config=player_detection_config
    )

    pose_sequence_quality_report = assess_pose_sequence_quality(
        clip_player_pose_sequences=clip_player_pose_sequence,
        config=player_detection_config
    )


    return (clip_player_pose_sequence, pose_sequence_quality_report)