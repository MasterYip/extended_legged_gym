from dataclasses import dataclass


@dataclass
class LidarConfig:
    """Minimal grid LiDAR configuration."""

    update_frequency: float = 50.0
    max_range: float = 50.0
    min_range: float = 0.2

    horizontal_line_num: int = 80
    vertical_line_num: int = 50
    horizontal_fov_deg_min: float = -180.0
    horizontal_fov_deg_max: float = 180.0
    vertical_fov_deg_min: float = -2.0
    vertical_fov_deg_max: float = 57.0

    pointcloud_in_world_frame: bool = False
    synchronize: bool = False
