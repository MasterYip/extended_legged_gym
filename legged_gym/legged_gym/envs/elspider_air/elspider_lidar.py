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
        """Initialize buffers including LiDAR observation buffers."""
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

    def post_physics_step(self):
        """Update after physics step, including LiDAR sensor."""
        # Update sensor pose before parent's post_physics_step
        self._update_sensor_pose()
        
        # Update LiDAR sensor
        self.lidar_update_time += self.dt
        if self.lidar_update_time >= self.lidar_update_interval:
            self._update_lidar()
            self.lidar_update_time = 0.0
        
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
        
        # Compute minimum obstacle distance
        valid_mask = self.lidar_dist_buf < self.lidar_cfg.max_range
        self.min_obstacle_dist[:] = self.lidar_cfg.max_range
        for i in range(self.num_envs):
            valid_dists = self.lidar_dist_buf[i][valid_mask[i]]
            if valid_dists.numel() > 0:
                self.min_obstacle_dist[i] = valid_dists.min()
        
        # Convert to spherical coordinates and downsample for observation
        sphere_points = cart2sphere(self.lidar_points_buf.view(-1, 3)).view(self.num_envs, -1, 3)
        downsampled = downsample_spherical_points_vectorized(
            sphere_points, self.num_theta_bins, self.num_phi_bins
        )
        
        # Use normalized distance as observation (0 = close, 1 = far/no hit)
        self.lidar_obs_buf[:] = downsampled[:, :, 0].clamp(0, self.lidar_cfg.max_range) / self.lidar_cfg.max_range

    def compute_observations(self):
        """Compute observations including LiDAR data."""
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
        self.obs_buf = torch.cat((base_obs, self.lidar_obs_buf), dim=-1)
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """Get noise scale vector including LiDAR observation noise."""
        # Get base noise vector
        noise_vec = super()._get_noise_scale_vec(cfg)
        
        # Extend for LiDAR observations
        num_lidar_obs = self.num_theta_bins * self.num_phi_bins
        
        # Add small noise to LiDAR observations
        if hasattr(cfg.noise.noise_scales, 'lidar'):
            noise_vec[-num_lidar_obs:] = cfg.noise.noise_scales.lidar * cfg.noise.noise_level
        else:
            noise_vec[-num_lidar_obs:] = 0.02 * cfg.noise.noise_level  # Default small noise
        
        return noise_vec

    def reset_idx(self, env_ids):
        """Reset environments including LiDAR sensor."""
        super().reset_idx(env_ids)
        
        # Reset LiDAR buffers for reset environments
        if len(env_ids) > 0:
            self.lidar_obs_buf[env_ids] = 0.0
            self.min_obstacle_dist[env_ids] = self.lidar_cfg.max_range
            
            # Reset LiDAR sensor for specific environments
            if hasattr(self, 'lidar_sensor'):
                self.lidar_sensor.reset(env_ids)

    def check_termination(self):
        """Check termination conditions including collision detection."""
        super().check_termination()
        
        # Get collision parameters from config
        if hasattr(self.cfg.rewards, 'collision_threshold'):
            collision_threshold = self.cfg.rewards.collision_threshold
        else:
            collision_threshold = 0.08  # Default 8cm - more permissive
        
        # Get protection steps (grace period during early training)
        if hasattr(self.cfg.rewards, 'collision_termination_after_steps'):
            min_steps = self.cfg.rewards.collision_termination_after_steps
        else:
            min_steps = 10  # Default: only check collision after 10 steps
        
        # Only terminate due to collision after protection period
        # This allows the robot to learn without being immediately terminated
        collision = self.min_obstacle_dist < collision_threshold
        collision_termination = collision & (self.episode_length_buf > min_steps)
        self.reset_buf |= collision_termination

    # ============== Reward Functions ==============
    
    def _reward_obstacle_avoidance(self):
        """Reward for maintaining safe distance from obstacles."""
        # Reward increases with distance from obstacles
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Compute reward based on minimum distance
        dist_reward = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        return dist_reward

    def _reward_collision_penalty(self):
        """Penalty for getting too close to obstacles."""
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.3)
        
        # Exponential penalty for being too close
        penalty = torch.exp(-self.min_obstacle_dist / danger_dist + 1) - 1
        penalty = torch.clamp(penalty, 0, 10)
        return -penalty

    def _reward_exploration(self):
        """Reward for exploring while avoiding obstacles."""
        # Combine forward velocity with obstacle avoidance
        forward_vel = self.base_lin_vel[:, 0]
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Only reward forward movement when it's safe
        safety_factor = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        exploration_reward = forward_vel * safety_factor
        return torch.clamp(exploration_reward, -1, 1)

    def _draw_debug_vis(self):
        """Draw debug visualization including LiDAR points."""
        super()._draw_debug_vis()
        
        # Draw LiDAR points for first environment
        if not self.headless and hasattr(self, 'lidar_points_buf'):
            self._draw_lidar_points()

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
