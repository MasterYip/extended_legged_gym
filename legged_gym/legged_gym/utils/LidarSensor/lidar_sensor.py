import math

import torch
import warp as wp

from .sensor_config.lidar_sensor_config import LidarConfig
from .sensor_kernels.lidar_kernels_warp import LidarWarpKernels


class LidarSensor:
    """Minimal Warp-based grid LiDAR sensor."""

    def __init__(
        self,
        env,
        env_cfg=None,
        sensor_config: LidarConfig = None,
        num_sensors=1,
        device="cuda:0",
    ):
        self.env = env
        self.env_cfg = env_cfg
        self.sensor_cfg = sensor_config or LidarConfig()
        self.num_sensors = num_sensors
        self.device = device

        self.num_envs = self.env["num_envs"]
        self.mesh_ids = self.env["mesh_ids"]
        self.lidar_positions_tensor = self.env["sensor_pos_tensor"]
        self.lidar_quat_tensor = self.env["sensor_quat_tensor"]
        self.far_plane = self.sensor_cfg.max_range
        self.pointcloud_in_world_frame = self.sensor_cfg.pointcloud_in_world_frame
        self.synchronize = self.sensor_cfg.synchronize

        if self.lidar_positions_tensor is None or self.lidar_quat_tensor is None:
            raise ValueError("LidarSensor requires sensor_pos_tensor and sensor_quat_tensor.")

        wp.init()
        self._init_ray_vectors()
        self._init_tensors()

    def _init_ray_vectors(self):
        self.num_vertical_lines = self.sensor_cfg.vertical_line_num
        self.num_horizontal_lines = self.sensor_cfg.horizontal_line_num

        horizontal_min = math.radians(self.sensor_cfg.horizontal_fov_deg_min)
        horizontal_max = math.radians(self.sensor_cfg.horizontal_fov_deg_max)
        vertical_min = math.radians(self.sensor_cfg.vertical_fov_deg_min)
        vertical_max = math.radians(self.sensor_cfg.vertical_fov_deg_max)

        if horizontal_max - horizontal_min > 2.0 * math.pi:
            raise ValueError("Horizontal FOV must be <= 360 degrees.")
        if vertical_max - vertical_min > math.pi:
            raise ValueError("Vertical FOV must be <= 180 degrees.")

        h_den = max(self.num_horizontal_lines - 1, 1)
        v_den = max(self.num_vertical_lines - 1, 1)
        ray_vectors = torch.zeros(
            (self.num_vertical_lines, self.num_horizontal_lines, 3),
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(self.num_vertical_lines):
            elevation = vertical_max - (vertical_max - vertical_min) * (i / v_den)
            for j in range(self.num_horizontal_lines):
                azimuth = horizontal_max - (horizontal_max - horizontal_min) * (j / h_den)
                ray_vectors[i, j, 0] = math.cos(azimuth) * math.cos(elevation)
                ray_vectors[i, j, 1] = math.sin(azimuth) * math.cos(elevation)
                ray_vectors[i, j, 2] = math.sin(elevation)

        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def _init_tensors(self):
        self.lidar_positions = wp.from_torch(
            self.lidar_positions_tensor.reshape(self.num_envs, self.num_sensors, 3),
            dtype=wp.vec3,
        )
        self.lidar_quat_array = wp.from_torch(
            self.lidar_quat_tensor.reshape(self.num_envs, self.num_sensors, 4),
            dtype=wp.quat,
        )

        self.lidar_tensor = torch.zeros(
            (
                self.num_envs,
                self.num_sensors,
                self.num_vertical_lines,
                self.num_horizontal_lines,
                3,
            ),
            device=self.device,
        )
        self.lidar_dist_tensor = torch.zeros(
            (
                self.num_envs,
                self.num_sensors,
                self.num_vertical_lines,
                self.num_horizontal_lines,
            ),
            device=self.device,
        )

        self.lidar_warp_tensor = wp.from_torch(self.lidar_tensor, dtype=wp.vec3)
        self.local_dist = wp.from_torch(self.lidar_dist_tensor, dtype=wp.float32)

    def reset(self, env_ids=None, value=0.0):
        if env_ids is None:
            self.lidar_tensor.fill_(value)
            self.lidar_dist_tensor.fill_(value)
            return

        self.lidar_tensor[env_ids].fill_(value)
        self.lidar_dist_tensor[env_ids].fill_(value)

    def update(self):
        wp.launch(
            kernel=LidarWarpKernels.draw_pointcloud,
            dim=(
                self.num_envs,
                self.num_sensors,
                self.num_vertical_lines,
                self.num_horizontal_lines,
            ),
            inputs=[
                self.mesh_ids,
                self.lidar_positions,
                self.lidar_quat_array,
                self.ray_vectors,
                self.far_plane,
                self.lidar_warp_tensor,
                self.local_dist,
                self.pointcloud_in_world_frame,
            ],
            device=self.device,
        )
        if self.synchronize:
            wp.synchronize(device=self.device)

        return self.lidar_tensor, self.lidar_dist_tensor
