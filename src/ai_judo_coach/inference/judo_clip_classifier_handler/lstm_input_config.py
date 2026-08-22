from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LSTMInputConfig:
    sequence_length: int = 210
    keypoint_count: int = 17
    coordinates_per_keypoint: int = 2
    unresolved_coordinate_value: float = 0.0

    @property
    def features_per_player(self) -> int:
        return self.keypoint_count * self.coordinates_per_keypoint

    @property
    def total_feature_count(self) -> int:
        return 2 * self.features_per_player
