from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class PersonDetection:
    """
    Stores data for a single pose detected by yolo 
    in a frame.
    """

    detection_idx: int   # index of the person detection within the given frame
    track_id: int | None  # temporary id assigned to this pose by bytetrack

    bbox_xyxy_px: np.ndarray  # raw coords (not normalized)
    bbox_xyxy_normalized: np.ndarray
    bbox_conf: float

    keypoints_xy_px: np.ndarray
    keypoints_xy_norm: np.ndarray
    keypoints_conf: np.ndarray

    def __post_init__(self) -> None:
        """
        Runs automatically after the dataclass is created.

        Normalises types and validates expected YOLO pose shapes.
        """

        self.detection_idx = int(self.detection_idx)

        if self.track_id is not None:
            self.track_id = int(self.track_id)

        self.bbox_conf = float(self.bbox_conf)

        self.bbox_xyxy_px = np.asarray(
            self.bbox_xyxy_px,
            dtype=np.float32,
        )

        self.keypoints_xy_px = np.asarray(
            self.keypoints_xy_px,
            dtype=np.float32,
        )

        self.keypoints_xy_norm = np.asarray(
            self.keypoints_xy_norm,
            dtype=np.float32,
        )

        self.keypoints_conf = np.asarray(
            self.keypoints_conf,
            dtype=np.float32,
        )

        if self.bbox_xyxy_px.shape != (4,):
            raise ValueError(
                f"bbox_xyxy_px must have shape (4,), "
                f"got {self.bbox_xyxy_px.shape}"
            )

        if self.keypoints_xy_px.shape != (17, 2):
            raise ValueError(
                f"keypoints_xy_px must have shape (17, 2), "
                f"got {self.keypoints_xy_px.shape}"
            )

        if self.keypoints_xy_norm.shape != (17, 2):
            raise ValueError(
                f"keypoints_xy_norm must have shape (17, 2), "
                f"got {self.keypoints_xy_norm.shape}"
            )

        if self.keypoints_conf.shape != (17,):
            raise ValueError(
                f"keypoints_conf must have shape (17,), "
                f"got {self.keypoints_conf.shape}"
            )

        x1, y1, x2, y2 = self.bbox_xyxy_px

        if x2 < x1 or y2 < y1:
            raise ValueError(
                f"Invalid bbox_xyxy_px for detection_idx={self.detection_idx}: "
                f"{self.bbox_xyxy_px}"
            )


@dataclass(slots=True)
class FrameDetections:
    """
    Stores meta data and a reference to 
    all PersonDetection instances within a given frame.
    Stores these references as a list
    """

    person_detections: list[PersonDetection]
    frame_idx: int

    frame_shape_hw: tuple[int, int]   # height and width of frame in pixels

    def __post_init__(self) -> None:
        """
        Runs automatically after the dataclass is created.

        Normalises frame metadata and validates contained detections.
        """

        self.frame_idx = int(self.frame_idx)
        self.person_detections = list(self.person_detections)

        if len(self.frame_shape_hw) != 2:
            raise ValueError(
                f"frame_shape_hw must have length 2, got {self.frame_shape_hw}"
            )

        frame_height = int(self.frame_shape_hw[0])
        frame_width = int(self.frame_shape_hw[1])

        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(
                f"frame_shape_hw must contain positive values, "
                f"got {(frame_height, frame_width)}"
            )

        self.frame_shape_hw = (frame_height, frame_width)

        for detection in self.person_detections:
            if not isinstance(detection, PersonDetection):
                raise TypeError(
                    "person_detections must only contain PersonDetection "
                    f"instances, got {type(detection)}"
                )


@dataclass(slots=True)
class ClipDetections:
    """
    Stores a list of all FrameDetections instances
    for a given input clip
    """

    frame_detections: list[FrameDetections]
    clip_id: str

    def __post_init__(self) -> None:
        """
        Runs automatically after the dataclass is created.

        Normalises clip metadata and validates contained frames.
        """

        self.clip_id = str(self.clip_id)
        self.frame_detections = list(self.frame_detections)

        for frame_detection in self.frame_detections:
            if not isinstance(frame_detection, FrameDetections):
                raise TypeError(
                    "frame_detections must only contain FrameDetections "
                    f"instances, got {type(frame_detection)}"
                )
