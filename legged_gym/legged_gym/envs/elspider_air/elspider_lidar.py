# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Extended for LiDAR-based confined space navigation

"""
ElSpider LiDAR Environment for Confined Space Navigation
基于激光雷达的六足机器人受限空间强化学习避障运动控制
"""

from time import time
import numpy as np
import os
import sys

from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate_inverse
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
import warp as wp
import trimesh

from legged_gym.envs.elspider_air.elspider import ElSpider
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw

# Import LidarSensor from OmniPerception
try:
    from LidarSensor.lidar_sensor import LidarSensor
    from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
except ImportError:
    # Add OmniPerception to path if not installed
    omni_perception_path = os.path.join(os.path.dirname(LEGGED_GYM_ROOT_DIR), '..', 'OmniPerception', 'LidarSensor')
    if os.path.exists(omni_perception_path):
        sys.path.insert(0, omni_perception_path)
        from LidarSensor.lidar_sensor import LidarSensor
        from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
    else:
        raise ImportError("Cannot find LidarSensor module. Please ensure OmniPerception is properly installed.")


@torch.jit.script
def quat_from_euler_xyz_tensor(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert euler angles to quaternion (tensor version)"""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def cart2sphere(cart: torch.Tensor) -> torch.Tensor:
    """Convert cartesian coordinates to spherical (r, theta, phi)"""
    epsilon = 1e-9
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    r = torch.norm(cart, dim=1)
    theta = torch.atan2(y, x)
    phi = torch.asin(z / (r + epsilon))
    return torch.stack((r, theta, phi), dim=-1)


def downsample_spherical_points_vectorized(sphere_points: torch.Tensor, 
                                           num_theta_bins: int = 10, 
                                           num_phi_bins: int = 10) -> torch.Tensor:
    """Downsample spherical points using bin averaging"""
    num_envs = sphere_points.shape[0]
    device = sphere_points.device
    num_bins = num_theta_bins * num_phi_bins
    
    theta_min, theta_max = -3.14, 3.14
    phi_min, phi_max = -0.5, 0.5  # Adjusted for typical LiDAR FOV
    
    r = sphere_points[:, :, 0]
    theta = sphere_points[:, :, 1]
    phi = sphere_points[:, :, 2]
    
    theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
    phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()
    theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
    phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)
    bin_indices = theta_bin * num_phi_bins + phi_bin
    
    r_sum = torch.zeros(num_envs, num_bins, device=device)
    bin_count = torch.zeros(num_envs, num_bins, device=device)
    r_sum.scatter_add_(1, bin_indices, r)
    ones = torch.ones_like(r)
    bin_count.scatter_add_(1, bin_indices, ones)
    bin_count = torch.clamp(bin_count, min=1.0)
    avg_r = r_sum / bin_count
    
    theta_centers = torch.linspace(
        theta_min + (theta_max - theta_min) / (2 * num_theta_bins),
        theta_max - (theta_max - theta_min) / (2 * num_theta_bins),
        num_theta_bins, device=device
    )
    phi_centers = torch.linspace(
        phi_min + (phi_max - phi_min) / (2 * num_phi_bins),
        phi_max - (phi_max - phi_min) / (2 * num_phi_bins),
        num_phi_bins, device=device
    )
    theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
    theta_centers_flat = theta_grid.reshape(-1)
    phi_centers_flat = phi_grid.reshape(-1)
    
    downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
    downsampled[:, :, 0] = avg_r
    downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0)
    downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)
    return downsampled


class ElSpiderLidar(ElSpider):
    """ElSpider robot with LiDAR sensor for confined space navigation.
    
    This class extends ElSpider to include:
    - LiDAR sensor integration using OmniPerception
    - LiDAR-based observations for obstacle avoidance
    - Rewards for collision avoidance in confined spaces
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Initialize LiDAR configuration before calling parent __init__
        self._init_lidar_cfg(cfg)
        
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize LiDAR sensor after environment is created
        self._init_lidar_sensor()
        
        print(f"[ElSpiderLidar] Initialized with LiDAR sensor")
        print(f"  - LiDAR type: {self.lidar_cfg.sensor_type}")
        print(f"  - LiDAR obs dim: {self.cfg.env.num_lidar_obs}")
        print(f"  - Total obs dim: {self.cfg.env.num_observations}")
        print(f"  - Goal navigation: {getattr(self.cfg.terrain, 'goal_navigation', False)}")
        if getattr(self.cfg.terrain, 'goal_navigation', False):
            print(f"  - Goal offset Y: {getattr(self.cfg.terrain, 'goal_offset_y', 4.0)}m")

    def _init_lidar_cfg(self, cfg):
        """Initialize LiDAR configuration from environment config."""
        self.lidar_cfg = LidarConfig()
        
        # Set LiDAR type from config if specified
        if hasattr(cfg, 'lidar') and hasattr(cfg.lidar, 'sensor_type'):
            self.lidar_cfg.sensor_type = cfg.lidar.sensor_type
        else:
            # Default to simple grid for lower computational cost
            self.lidar_cfg.sensor_type = LidarType.SIMPLE_GRID
        
        # Configure LiDAR parameters
        if hasattr(cfg, 'lidar'):
            lidar_cfg = cfg.lidar
            self.lidar_cfg.update_frequency = getattr(lidar_cfg, 'update_frequency', 20.0)
            self.lidar_cfg.max_range = getattr(lidar_cfg, 'max_range', 5.0)
            self.lidar_cfg.min_range = getattr(lidar_cfg, 'min_range', 0.1)
            self.lidar_cfg.horizontal_line_num = getattr(lidar_cfg, 'horizontal_line_num', 36)
            self.lidar_cfg.vertical_line_num = getattr(lidar_cfg, 'vertical_line_num', 10)
            self.lidar_cfg.horizontal_fov_deg_min = getattr(lidar_cfg, 'horizontal_fov_deg_min', -180)
            self.lidar_cfg.horizontal_fov_deg_max = getattr(lidar_cfg, 'horizontal_fov_deg_max', 180)
            self.lidar_cfg.vertical_fov_deg_min = getattr(lidar_cfg, 'vertical_fov_deg_min', -15)
            self.lidar_cfg.vertical_fov_deg_max = getattr(lidar_cfg, 'vertical_fov_deg_max', 15)
            
            # LiDAR observation configuration
            self.num_theta_bins = getattr(lidar_cfg, 'num_theta_bins', 12)
            self.num_phi_bins = getattr(lidar_cfg, 'num_phi_bins', 8)
        else:
            # Default configuration for grid LiDAR
            self.lidar_cfg.update_frequency = 20.0
            self.lidar_cfg.max_range = 5.0
            self.lidar_cfg.min_range = 0.1
            self.lidar_cfg.horizontal_line_num = 36
            self.lidar_cfg.vertical_line_num = 10
            self.lidar_cfg.horizontal_fov_deg_min = -180
            self.lidar_cfg.horizontal_fov_deg_max = 180
            self.lidar_cfg.vertical_fov_deg_min = -15
            self.lidar_cfg.vertical_fov_deg_max = 15
            self.num_theta_bins = 12
            self.num_phi_bins = 8
        
        # Set dt to match simulation
        self.lidar_cfg.dt = 0.02  # Will be updated in init
        self.lidar_cfg.pointcloud_in_world_frame = False  # Local frame for observations

    def _init_lidar_sensor(self):
        """Initialize the LiDAR sensor after simulation is created."""
        # Update dt from simulation
        self.lidar_cfg.dt = self.dt
        
        # Create WARP mesh from terrain
        self._create_warp_mesh()
        
        # Create sensor tensor dictionary
        self.warp_tensor_dict = self._create_warp_tensor_dict()
        
        # Initialize LiDAR sensor
        self.lidar_sensor = LidarSensor(
            env=self.warp_tensor_dict,
            env_cfg=None,
            sensor_config=self.lidar_cfg,
            num_sensors=1,
            device=self.device
        )
        
        # Capture render graph for faster execution
        self.lidar_sensor.capture()
        
        # Initialize timing
        self.lidar_update_time = 0.0
        self.lidar_update_interval = 1.0 / self.lidar_cfg.update_frequency

    def _create_warp_mesh(self):
        """Create WARP mesh from terrain for ray casting."""
        wp.init()
        
        # Get terrain mesh vertices and triangles
        if hasattr(self, 'terrain') and self.terrain is not None:
            if hasattr(self.terrain, 'vertices') and self.terrain.vertices is not None:
                vertices = self.terrain.vertices.copy()
                triangles = self.terrain.triangles.copy()
                
                # Apply terrain offset
                if hasattr(self.cfg.terrain, 'border_size'):
                    vertices[:, 0] -= self.cfg.terrain.border_size
                    vertices[:, 1] -= self.cfg.terrain.border_size
            else:
                # Create simple ground plane if no terrain mesh
                vertices = np.array([
                    [-50, -50, 0],
                    [50, -50, 0],
                    [50, 50, 0],
                    [-50, 50, 0]
                ], dtype=np.float32)
                triangles = np.array([
                    [0, 1, 2],
                    [0, 2, 3]
                ], dtype=np.int32)
        else:
            # Create simple ground plane
            vertices = np.array([
                [-50, -50, 0],
                [50, -50, 0],
                [50, 50, 0],
                [-50, 50, 0]
            ], dtype=np.float32)
            triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ], dtype=np.int32)
        
        # Convert to WARP arrays
        vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        vertex_wp = wp.from_torch(vertex_tensor, dtype=wp.vec3)
        faces_wp = wp.from_numpy(triangles.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        
        # Create WARP mesh
        self.wp_mesh = wp.Mesh(points=vertex_wp, indices=faces_wp)
        self.mesh_ids = wp.array([self.wp_mesh.id], dtype=wp.uint64)

    def _create_warp_tensor_dict(self):
        """Create tensor dictionary for LiDAR sensor."""
        warp_dict = {}
        
        # Sensor position and orientation tensors
        self.sensor_pos_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.sensor_quat_tensor = torch.zeros(self.num_envs, 4, device=self.device)
        
        # LiDAR mounting offset (on top of robot body)
        # Default: 0.15m above body center, facing forward
        if hasattr(self.cfg, 'lidar') and hasattr(self.cfg.lidar, 'sensor_offset'):
            self.sensor_translation_local = torch.tensor(
                self.cfg.lidar.sensor_offset, device=self.device
            )
        else:
            self.sensor_translation_local = torch.tensor([0.0, 0.0, 0.15], device=self.device)
        
        # Sensor rotation offset (no rotation by default)
        if hasattr(self.cfg, 'lidar') and hasattr(self.cfg.lidar, 'sensor_rotation_deg'):
            roll = np.deg2rad(self.cfg.lidar.sensor_rotation_deg[0])
            pitch = np.deg2rad(self.cfg.lidar.sensor_rotation_deg[1])
            yaw = np.deg2rad(self.cfg.lidar.sensor_rotation_deg[2])
            self.sensor_offset_quat_local = quat_from_euler_xyz_tensor(
                torch.tensor([roll], device=self.device),
                torch.tensor([pitch], device=self.device),
                torch.tensor([yaw], device=self.device)
            ).squeeze()
        else:
            self.sensor_offset_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        
        # Expand to all environments
        self.sensor_translation = self.sensor_translation_local.repeat(self.num_envs, 1)
        self.sensor_offset_quat = self.sensor_offset_quat_local.repeat(self.num_envs, 1)
        
        # Update initial sensor poses
        self._update_sensor_pose()
        
        # Populate dictionary
        warp_dict['device'] = self.device
        warp_dict['num_envs'] = self.num_envs
        warp_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        warp_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        warp_dict['mesh_ids'] = self.mesh_ids
        
        return warp_dict

    def _update_sensor_pose(self):
        """Update LiDAR sensor pose based on robot base pose."""
        # Compute sensor position in world frame
        self.sensor_pos_tensor[:] = self.root_states[:, :3] + quat_apply(
            self.root_states[:, 3:7], self.sensor_translation
        )
        
        # Compute sensor orientation in world frame
        self.sensor_quat_tensor[:] = quat_mul(
            self.root_states[:, 3:7], self.sensor_offset_quat
        )

    def _init_buffers(self):
        """Initialize buffers including LiDAR observation buffers and goal buffers."""
        # Set goal_navigation flag BEFORE super()._init_buffers()
        # because _get_noise_scale_vec is called inside super()._init_buffers()
        self.goal_navigation = getattr(self.cfg.terrain, 'goal_navigation', False)
        
        super()._init_buffers()
        
        # LiDAR observation buffers
        num_lidar_obs = self.num_theta_bins * self.num_phi_bins
        self.lidar_obs_buf = torch.zeros(
            self.num_envs, num_lidar_obs, device=self.device, requires_grad=False
        )
        
        # Raw LiDAR data buffers
        total_rays = self.lidar_cfg.horizontal_line_num * self.lidar_cfg.vertical_line_num
        self.lidar_points_buf = torch.zeros(
            self.num_envs, total_rays, 3, device=self.device, requires_grad=False
        )
        self.lidar_dist_buf = torch.zeros(
            self.num_envs, total_rays, device=self.device, requires_grad=False
        )
        
        # Minimum distance to obstacles (for rewards)
        self.min_obstacle_dist = torch.ones(
            self.num_envs, device=self.device, requires_grad=False
        ) * self.lidar_cfg.max_range
        
        # Goal navigation buffers
        if self.goal_navigation:
            self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)  # xyz goal
            self.goal_distance = torch.zeros(self.num_envs, device=self.device)
            self.prev_goal_distance = torch.zeros(self.num_envs, device=self.device)
            self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.goal_obs_buf = torch.zeros(self.num_envs, 2, device=self.device)  # [angle, dist]
            self.goal_offset_y = getattr(self.cfg.terrain, 'goal_offset_y', 4.0)

    def post_physics_step(self):
        """Update after physics step, including LiDAR sensor and goal tracking."""
        # Update sensor pose before parent's post_physics_step
        self._update_sensor_pose()
        
        # Update LiDAR sensor
        self.lidar_update_time += self.dt
        if self.lidar_update_time >= self.lidar_update_interval:
            self._update_lidar()
            self.lidar_update_time = 0.0
        
        # Update goal distance tracking (before rewards are computed)
        if self.goal_navigation:
            self._update_goal_tracking()
        
        # Call parent's post_physics_step
        super().post_physics_step()

    def _update_lidar(self):
        """Update LiDAR sensor and process observations."""
        # Get raw LiDAR data
        lidar_points, lidar_dist = self.lidar_sensor.update()
        
        # Reshape data: (num_envs, num_sensors, v_lines, h_lines, 3) -> (num_envs, total_rays, 3)
        total_rays = self.lidar_cfg.horizontal_line_num * self.lidar_cfg.vertical_line_num
        self.lidar_points_buf[:] = lidar_points.view(self.num_envs, -1, 3)[:, :total_rays, :]
        self.lidar_dist_buf[:] = lidar_dist.view(self.num_envs, -1)[:, :total_rays]
        
        # Compute minimum obstacle distance (vectorized, no Python loop)
        # Replace out-of-range values with max_range so they don't affect min
        clamped_dist = torch.where(
            self.lidar_dist_buf < self.lidar_cfg.max_range,
            self.lidar_dist_buf,
            torch.full_like(self.lidar_dist_buf, self.lidar_cfg.max_range)
        )
        self.min_obstacle_dist[:] = clamped_dist.min(dim=1)[0]
        
        # Convert to spherical coordinates and downsample for observation
        sphere_points = cart2sphere(self.lidar_points_buf.view(-1, 3)).view(self.num_envs, -1, 3)
        downsampled = downsample_spherical_points_vectorized(
            sphere_points, self.num_theta_bins, self.num_phi_bins
        )
        
        # Use normalized distance as observation (0 = close, 1 = far/no hit)
        self.lidar_obs_buf[:] = downsampled[:, :, 0].clamp(0, self.lidar_cfg.max_range) / self.lidar_cfg.max_range

    def compute_observations(self):
        """Compute observations including LiDAR data and goal information."""
        # Base observations (same as ElSpider)
        base_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        
        # Add height measurements if configured
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            base_obs = torch.cat((base_obs, heights), dim=-1)
        
        # Add LiDAR observations
        obs_parts = [base_obs, self.lidar_obs_buf]
        
        # Add goal observations if enabled
        if self.goal_navigation:
            obs_parts.append(self.goal_obs_buf)
        
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """Get noise scale vector including LiDAR and goal observation noise."""
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        
        # Base proprioception noise (same as ElSpider parent)
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.  # previous actions
        
        # Height measurements noise
        if self.cfg.terrain.measure_heights:
            noise_vec[66:253] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        
        # LiDAR observation noise
        num_lidar_obs = self.num_theta_bins * self.num_phi_bins
        lidar_noise = getattr(noise_scales, 'lidar', 0.05) * noise_level
        lidar_start = 253
        lidar_end = lidar_start + num_lidar_obs
        noise_vec[lidar_start:lidar_end] = lidar_noise
        
        # Goal observation noise (small)
        if self.goal_navigation:
            noise_vec[lidar_end:lidar_end+2] = 0.02 * noise_level
        
        return noise_vec

    def reset_idx(self, env_ids):
        """Reset environments including LiDAR sensor and goal positions."""
        # Log goal stats before reset (while data is still valid)
        if self.goal_navigation and len(env_ids) > 0 and self.init_done and "episode" in self.extras:
            goal_reached_ratio = self.goal_reached[env_ids].float().mean().item()
            mean_goal_dist = self.goal_distance[env_ids].mean().item()
            self.extras["episode"]["goal_reached_ratio"] = goal_reached_ratio
            self.extras["episode"]["mean_goal_distance"] = mean_goal_dist
        
        super().reset_idx(env_ids)
        
        # Reset LiDAR buffers for reset environments
        if len(env_ids) > 0:
            self.lidar_obs_buf[env_ids] = 0.0
            self.min_obstacle_dist[env_ids] = self.lidar_cfg.max_range
            
            # Reset LiDAR sensor for specific environments
            if hasattr(self, 'lidar_sensor'):
                self.lidar_sensor.reset(env_ids)
            
            # Set goal positions for reset environments
            if self.goal_navigation:
                self._set_goal_positions(env_ids)
    
    def _set_goal_positions(self, env_ids):
        """Set goal positions for given environments.
        Goal is placed at +Y offset from env_origin (far end of corridor).
        """
        # Goal position: same X as origin, offset Y forward, same Z as origin
        self.goal_positions[env_ids, 0] = self.env_origins[env_ids, 0]  # Same X
        self.goal_positions[env_ids, 1] = self.env_origins[env_ids, 1] + self.goal_offset_y  # Forward Y
        self.goal_positions[env_ids, 2] = self.env_origins[env_ids, 2]  # Same Z
        
        # Add small random variation to goal position (within corridor width)
        self.goal_positions[env_ids, 0] += torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        
        # Initialize goal distances
        self.goal_distance[env_ids] = torch.norm(
            self.root_states[env_ids, :2] - self.goal_positions[env_ids, :2], dim=1
        )
        self.prev_goal_distance[env_ids] = self.goal_distance[env_ids].clone()
        self.goal_reached[env_ids] = False
    
    def _update_goal_tracking(self):
        """Update goal distance and goal observations each step."""
        # Save previous distance
        self.prev_goal_distance[:] = self.goal_distance.clone()
        
        # Compute current distance to goal (XY plane)
        self.goal_distance[:] = torch.norm(
            self.root_states[:, :2] - self.goal_positions[:, :2], dim=1
        )
        
        # Check if goal reached
        goal_threshold = getattr(self.cfg.rewards, 'goal_reach_threshold', 1.0)
        self.goal_reached[:] = self.goal_distance < goal_threshold
        
        # Compute goal observation: [direction_angle, normalized_distance]
        # Direction to goal in robot's local frame
        goal_vec_world = self.goal_positions[:, :2] - self.root_states[:, :2]  # (N, 2)
        goal_vec_local = self._world_to_local_2d(goal_vec_world)  # (N, 2)
        
        # Angle to goal (in local frame): atan2(local_y, local_x)
        goal_angle = torch.atan2(goal_vec_local[:, 1], goal_vec_local[:, 0])  # (-pi, pi)
        
        # Normalized distance (0 = at goal, 1 = max distance)
        goal_max_dist = getattr(self.cfg.rewards, 'goal_max_distance', 8.0)
        goal_dist_normalized = torch.clamp(self.goal_distance / goal_max_dist, 0, 1)
        
        self.goal_obs_buf[:, 0] = goal_angle / 3.14159  # Normalize to [-1, 1]
        self.goal_obs_buf[:, 1] = goal_dist_normalized
    
    def _world_to_local_2d(self, vec_world):
        """Convert 2D world vectors to robot's local frame using yaw rotation."""
        # Get robot yaw from quaternion
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        
        # Rotate to local frame
        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)
        local_x = cos_yaw * vec_world[:, 0] - sin_yaw * vec_world[:, 1]
        local_y = sin_yaw * vec_world[:, 0] + cos_yaw * vec_world[:, 1]
        return torch.stack([local_x, local_y], dim=1)
    
    def _post_physics_step_callback(self):
        """Override to add goal-directed heading command."""
        # Resample commands on schedule
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)
                   == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        # Goal-directed heading: override heading command to point toward goal
        if self.goal_navigation and getattr(self.cfg.commands, 'goal_directed', False):
            # Compute heading angle from robot to goal
            goal_vec = self.goal_positions[:, :2] - self.root_states[:, :2]
            goal_heading = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
            
            # Set heading command to goal direction
            self.commands[:, 3] = goal_heading
            
            # Convert heading to angular velocity command (P controller)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_error = goal_heading - heading
            # Wrap to [-pi, pi]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            self.commands[:, 2] = torch.clip(0.8 * heading_error, -1., 1.)
            
            # Forward velocity: obstacle-aware goal speed
            # When facing goal: higher speed; near obstacles: automatically slow down
            max_vel = self.command_ranges["lin_vel_x"][1]  # 1.2 m/s
            min_vel = 0.1
            
            # facing_factor: 1.0 when facing goal, 0.0 when perpendicular, -1 when away
            cos_heading = torch.cos(heading_error)
            # Map cos_heading from [-1, 1] to [min_vel, max_vel]
            speed_factor = (cos_heading + 1.0) / 2.0  # 0~1
            goal_speed = min_vel + (max_vel - min_vel) * speed_factor

            # Obstacle-aware speed scaling using LiDAR minimum distance
            safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
            danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.15)
            obs_speed_scale = torch.clamp(
                (self.min_obstacle_dist - danger_dist) / (safe_dist - danger_dist + 1e-6),
                0.0, 1.0
            )
            self.commands[:, 0] = goal_speed * obs_speed_scale
            
            # Lateral velocity: small nudge to help navigate around obstacles
            self.commands[:, 1] = torch.clip(-0.3 * torch.sin(heading_error), -0.3, 0.3)
        else:
            # Default heading command conversion
            if self.cfg.commands.heading_command:
                forward = quat_apply(self.base_quat, self.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[:, 2] = torch.clip(0.5 * (self.commands[:, 3] - heading), -1., 1.)
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _update_terrain_curriculum(self, env_ids):
        """Override terrain curriculum for goal-directed confined navigation.
        
        Criteria:
        - Move UP: reached goal OR survived >50% AND walked >2m toward goal
        - Move DOWN: survived <15% of episode (died quickly)
        """
        if not self.init_done:
            return
        
        # Compute distance walked from spawn
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        
        # Survival ratio
        survival_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
        
        if self.goal_navigation:
            # Goal reached is the strongest signal for moving up
            reached_goal = self.goal_reached[env_ids]
            # Secondary: made significant progress toward goal (covered >60% of initial distance)
            initial_dist = self.goal_offset_y  # ~4.0m
            progress_ratio = 1.0 - self.goal_distance[env_ids] / (initial_dist + 1e-6)
            good_progress = (survival_ratio > 0.4) & (progress_ratio > 0.6)
            move_up = reached_goal | good_progress
        else:
            # Fallback: survival-based
            move_up = (survival_ratio > 0.4) & (distance > 1.0)
        
        move_down = (survival_ratio < 0.15) & (~move_up)
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def check_termination(self):
        """Check termination conditions including collision and goal reached."""
        super().check_termination()
        
        # Get collision parameters from config
        if hasattr(self.cfg.rewards, 'collision_threshold'):
            collision_threshold = self.cfg.rewards.collision_threshold
        else:
            collision_threshold = 0.08  # Default 8cm
        
        # Get protection steps
        if hasattr(self.cfg.rewards, 'collision_termination_after_steps'):
            min_steps = self.cfg.rewards.collision_termination_after_steps
        else:
            min_steps = 10
        
        # Terminate due to collision after protection period
        collision = self.min_obstacle_dist < collision_threshold
        collision_termination = collision & (self.episode_length_buf > min_steps)
        self.reset_buf |= collision_termination
        
        # Terminate (success) when goal is reached
        if self.goal_navigation:
            self.reset_buf |= self.goal_reached

    # ============== Reward Functions ==============
    
    def _reward_obstacle_avoidance(self):
        """Reward for maintaining safe distance from obstacles."""
        # Reward increases with distance from obstacles
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Compute reward based on minimum distance
        dist_reward = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        return dist_reward

    def _reward_collision_penalty(self):
        """Penalty for getting too close to obstacles.
        Returns positive value (higher = closer to obstacle).
        Should be used with NEGATIVE reward scale.
        """
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.3)
        
        # Exponential penalty for being too close
        # Returns positive value: large when close, ~0 when far
        penalty = torch.exp(-self.min_obstacle_dist / danger_dist + 1) - 1
        penalty = torch.clamp(penalty, 0, 10)
        return penalty

    def _reward_exploration(self):
        """Reward for exploring (moving) while avoiding obstacles."""
        # Reward any movement (not just forward) when safe from obstacles
        move_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Scale movement reward by safety — encourage moving only when safe
        safety_factor = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        
        # Also consider command tracking: reward moving in commanded direction
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        has_command = (cmd_speed > 0.1).float()
        exploration_reward = move_speed * safety_factor * has_command
        return torch.clamp(exploration_reward, 0, 1)

    def _reward_goal_reaching(self):
        """Reward for moving toward goal: velocity projected onto goal direction.
        This directly rewards the component of robot velocity pointing toward the goal.
        
        IMPORTANT: goal_dir is in WORLD frame, so we must use WORLD-frame velocity
        (root_states[:, 7:9]) NOT local-frame base_lin_vel.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Direction to goal (unit vector in XY plane, WORLD frame)
        goal_vec = self.goal_positions[:, :2] - self.root_states[:, :2]
        goal_dir = goal_vec / (torch.norm(goal_vec, dim=1, keepdim=True) + 1e-6)
        
        # Robot velocity in WORLD frame (root_states[:, 7:9] = world linear vel XY)
        world_vel_xy = self.root_states[:, 7:9]
        
        # Project world-frame velocity onto world-frame goal direction
        # Positive = moving toward goal, negative = moving away
        vel_toward_goal = torch.sum(world_vel_xy * goal_dir, dim=1)
        
        # Reward: proportional to velocity toward goal, capped at command speed
        reward = torch.clamp(vel_toward_goal, -0.5, 1.5)
        
        # Bonus multiplier when close to goal (last 2m): encourage final approach
        close_bonus = torch.where(
            self.goal_distance < 2.0,
            1.0 + (2.0 - self.goal_distance) * 0.5,  # up to 2x reward when very close
            torch.ones_like(self.goal_distance)
        )
        
        return reward * close_bonus

    def _reward_goal_progress(self):
        """Dense reward for reducing distance to goal each step.
        Positive when approaching goal, negative when moving away.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)

        progress = self.prev_goal_distance - self.goal_distance
        return torch.clamp(progress, -0.05, 0.10)
    
    def _reward_goal_bonus(self):
        """Large bonus reward for reaching the goal."""
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)
        
        return self.goal_reached.float()
    
    def _reward_goal_heading(self):
        """Reward for facing toward goal AND moving forward.
        Only rewards heading alignment when the robot is actually walking.
        Prevents 'face goal and stand still' local optimum.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Compute angle between robot forward and goal direction
        goal_vec = self.goal_positions[:, :2] - self.root_states[:, :2]
        goal_dir = goal_vec / (torch.norm(goal_vec, dim=1, keepdim=True) + 1e-6)
        
        forward = quat_apply(self.base_quat, self.forward_vec)
        forward_2d = forward[:, :2]
        forward_2d = forward_2d / (torch.norm(forward_2d, dim=1, keepdim=True) + 1e-6)
        
        # Dot product: 1 when facing goal, -1 when facing away
        cos_angle = torch.sum(forward_2d * goal_dir, dim=1)
        heading_reward = torch.clamp((cos_angle + 1.0) / 2.0, 0, 1)
        
        # Gate by movement speed: only reward heading when robot is walking
        # Use WORLD-frame velocity for consistency (root_states[:, 7:9])
        move_speed = torch.norm(self.root_states[:, 7:9], dim=1)
        moving_mask = torch.clamp(move_speed / 0.3, 0, 1)  # ramp: 0 at 0m/s, 1 at 0.3m/s+
        
        return heading_reward * moving_mask

    def _draw_debug_vis(self):
        """Draw debug visualization including LiDAR points and goal markers."""
        super()._draw_debug_vis()
        
        # Draw LiDAR points for first environment
        if not self.headless and hasattr(self, 'lidar_points_buf'):
            self._draw_lidar_points()
        
        # Draw goal markers
        if not self.headless and self.goal_navigation:
            self._draw_goal_markers()
    
    def _draw_goal_markers(self):
        """Visualize goal positions as red spheres."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
        
        # Draw goal for first few environments
        for env_idx in range(min(4, self.num_envs)):
            goal_pos = self.goal_positions[env_idx].cpu().numpy()
            sphere = gymutil.WireframeSphereGeometry(0.15, 8, 8, None, color=(1, 0, 0))
            pose = gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2] + 0.3), r=None)
            gymutil.draw_lines(sphere, self.gym, self.viewer, self.envs[env_idx], pose)

    def _draw_lidar_points(self):
        """Visualize LiDAR point cloud."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
        
        # Only draw for selected environment
        env_idx = 0
        points = self.lidar_points_buf[env_idx]
        sensor_pos = self.sensor_pos_tensor[env_idx]
        sensor_quat = self.sensor_quat_tensor[env_idx]
        
        # Transform points to world frame
        world_points = sensor_pos + quat_apply(sensor_quat.unsqueeze(0), points)
        
        # Draw subset of points (for performance)
        step = max(1, points.shape[0] // 100)
        for i in range(0, points.shape[0], step):
            pos = world_points[i].cpu().numpy()
            sphere = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
            pose = gymapi.Transform(gymapi.Vec3(*pos), r=None)
            gymutil.draw_lines(sphere, self.gym, self.viewer, self.envs[env_idx], pose)
