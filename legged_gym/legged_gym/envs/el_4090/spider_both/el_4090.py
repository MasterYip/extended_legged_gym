# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os
import warp as wp
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.elspider_air.elspider import ElSpider
from legged_gym.utils.LidarSensor import LidarConfig, LidarSensor

from legged_gym.envs.el_4090.spider_both.el4090_spider_config import El4090BothCfg,El4090BothCfgPPO


class EL_4090_BOTH(ElSpider):
    cfg : El4090BothCfg
     # env Init
    def __init__(self, cfg: El4090BothCfg, sim_params, physics_engine, sim_device, headless,task_name="el4090_both"):
        """ Parses the provided config file,Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.action_des = 0
        # self.action_flag = 1

        # 初始化 group1_contact_time 和 group2_contact_time
        self.group1_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.group2_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.T = 0.5
        
        # Debug counters
        self.debug_step_counter = 0

        

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False) 
        self.command_lin_vel_scale = torch.tensor(self.obs_scales.lin_vel,device=self.device, requires_grad=False,)
        self.command_ang_vel_scale = torch.tensor(self.obs_scales.ang_vel,device=self.device,requires_grad=False,)
        self.command_embedded_state_scale = torch.tensor(self.obs_scales.embedded_state,device=self.device,requires_grad=False,)
        
        mammal_joint_angles = getattr(
            self.cfg.init_state,
            "mammal_default_joint_angles",
            self.cfg.init_state.default_joint_angles,
        )
        self.mammal_default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i, name in enumerate(self.dof_names):
            self.mammal_default_dof_pos[i] = mammal_joint_angles[name]
        self.mammal_default_dof_pos = self.mammal_default_dof_pos.unsqueeze(0)
        self.haa_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if name.endswith("_HAA")],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.hfe_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if name.endswith("_HFE")],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.mammal_thigh_indices = torch.cat((self.haa_indices, self.hfe_indices), dim=0)
        self.embedded_state_default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.filtered_embedded_state_default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.embedded_state_transition_time = torch.zeros(
            (self.num_envs,),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )


        # 定义两组三角步态的脚部索引
        # 组1: LF (1), LB (0), RM (5)
        # 组2: RF (4), RB (3), LM (2)
        self.tripod_group1_indices = torch.tensor([1, 0, 5], device=self.device, dtype=torch.long)
        self.tripod_group2_indices = torch.tensor([4, 3, 2], device=self.device, dtype=torch.long)

        feet_names = getattr(self, "feet_names", [f"foot_{i}" for i in range(len(self.feet_indices))])
        group1_names = [feet_names[i] for i in self.tripod_group1_indices.cpu().tolist()]
        group2_names = [feet_names[i] for i in self.tripod_group2_indices.cpu().tolist()]
        print("[EL_4090 gait check]")
        print("  feet_names:", feet_names)
        print("  feet_indices:", self.feet_indices.detach().cpu().tolist())
        print("  tripod_group1_indices:", self.tripod_group1_indices.detach().cpu().tolist(), "names:", group1_names)
        print("  tripod_group2_indices:", self.tripod_group2_indices.detach().cpu().tolist(), "names:", group2_names)
        self.feet_first_contact = torch.zeros_like(self.feet_air_time, dtype=torch.bool)
        self.feet_air_time_on_contact = torch.zeros_like(self.feet_air_time)
        self.episode_base_height_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_base_height_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 初始化小腿索引 (SHANK)
        # Bodies:  ['base_link', 'LB_HIP', 'LB_THIGH', 'LB_SHANK', 'LB_FOOT', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 
        # 'LM_HIP', 'LM_THIGH', 'LM_SHANK', 'LM_FOOT', 'RB_HIP', 'RB_THIGH', 'RB_SHANK', 'RB_FOOT', 
        # 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RM_HIP', 'RM_THIGH', 'RM_SHANK', 'RM_FOOT']
        # 小腿顺序: LB_SHANK(3), LF_SHANK(7), LM_SHANK(11), RB_SHANK(15), RF_SHANK(19), RM_SHANK(23)
        shank_names = ['LB_SHANK', 'LF_SHANK', 'LM_SHANK', 'RB_SHANK', 'RF_SHANK', 'RM_SHANK']
        self.shank_indices = torch.zeros(len(shank_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shank_names)):
            self.shank_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], shank_names[i])

        if getattr(self.cfg.lidar, "enable", False):
            self._init_lidar()

    def _get_noise_scale_vec(self, cfg):
        """Observation noise layout: 67 proprio/command/action dims + lidar distances."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:13] = 0.0  # commands, including embedded state
        noise_vec[13:31] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[31:49] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[49:67] = 0.0  # previous actions

        if getattr(self.cfg.lidar, "enable", False):
            lidar_noise = getattr(noise_scales, "lidar", 0.0)
            noise_vec[67:] = lidar_noise * noise_level * self.obs_scales.lidar
        elif self.cfg.terrain.measure_heights:
            noise_vec[67:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_lidar(self):
        self.num_lidar_observations = self.cfg.lidar.vertical_line_num * self.cfg.lidar.horizontal_line_num
        self.lidar_distances = torch.zeros(
            (self.num_envs, self.num_lidar_observations),
            dtype=torch.float,
            device=self.device,
        )
        self.lidar_points = torch.zeros(
            self.num_envs,
            self.cfg.lidar.num_sensors,
            self.cfg.lidar.vertical_line_num,
            self.cfg.lidar.horizontal_line_num,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.lidar_sensor_pos_tensor = torch.zeros(self.num_envs, self.cfg.lidar.num_sensors, 3, device=self.device)
        self.lidar_sensor_quat_tensor = torch.zeros(self.num_envs, self.cfg.lidar.num_sensors, 4, device=self.device)
        self.lidar_position_offset = torch.tensor(self.cfg.lidar.position, dtype=torch.float, device=self.device)
        lidar_euler = torch.tensor(self.cfg.lidar.orientation_euler_deg, dtype=torch.float, device=self.device) * np.pi / 180.0
        self.lidar_quat_offset = quat_from_euler_xyz(
            lidar_euler[0].repeat(self.num_envs),
            lidar_euler[1].repeat(self.num_envs),
            lidar_euler[2].repeat(self.num_envs),
        )
        self.lidar_ray_vectors = self._build_lidar_ray_vectors()

        plane_mode = getattr(self.cfg.lidar, "plane_mode", None)
        if plane_mode is None:
            plane_mode = "zero" if getattr(self.cfg.lidar, "zero_on_plane", False) else "raycast"
        self.lidar_plane_mode = plane_mode
        self.lidar_zero_obs = self.cfg.terrain.mesh_type == "plane" and plane_mode == "zero"
        self.lidar_plane_obs = self.cfg.terrain.mesh_type == "plane" and plane_mode in ("raycast", "max_range", "zero")
        if self.lidar_plane_obs:
            self.lidar_sensor = None
            self._update_lidar()
            return

        if not hasattr(self, "terrain") or not hasattr(self.terrain, "vertices") or not hasattr(self.terrain, "triangles"):
            raise ValueError("LiDAR mode requires terrain.mesh_type='trimesh' so a raycast mesh is available.")

        vertices = self.terrain.vertices.copy().astype(np.float32)
        vertices[:, 0] -= self.cfg.terrain.border_size
        vertices[:, 1] -= self.cfg.terrain.border_size
        triangles = self.terrain.triangles.astype(np.int32).flatten()

        self.lidar_mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=self.device),
            indices=wp.array(triangles, dtype=wp.int32, device=self.device),
        )
        self.lidar_mesh_ids = wp.array([self.lidar_mesh.id], dtype=wp.uint64, device=self.device)

        lidar_cfg = LidarConfig(
            sensor_type="simple_grid",
            max_range=self.cfg.lidar.max_range,
            min_range=self.cfg.lidar.min_range,
            horizontal_line_num=self.cfg.lidar.horizontal_line_num,
            vertical_line_num=self.cfg.lidar.vertical_line_num,
            horizontal_fov_deg_min=self.cfg.lidar.horizontal_fov_deg_min,
            horizontal_fov_deg_max=self.cfg.lidar.horizontal_fov_deg_max,
            vertical_fov_deg_min=self.cfg.lidar.vertical_fov_deg_min,
            vertical_fov_deg_max=self.cfg.lidar.vertical_fov_deg_max,
            pointcloud_in_world_frame=self.cfg.lidar.pointcloud_in_world_frame,
        )

        self.lidar_env = {
            "num_envs": self.num_envs,
            "mesh_ids": self.lidar_mesh_ids,
            "sensor_pos_tensor": self.lidar_sensor_pos_tensor,
            "sensor_quat_tensor": self.lidar_sensor_quat_tensor,
        }
        self.lidar_sensor = LidarSensor(
            self.lidar_env,
            self.cfg,
            lidar_cfg,
            num_sensors=self.cfg.lidar.num_sensors,
            device=self.device,
        )
        self._update_lidar()

    def _build_lidar_ray_vectors(self):
        horizontal_min = np.deg2rad(self.cfg.lidar.horizontal_fov_deg_min)
        horizontal_max = np.deg2rad(self.cfg.lidar.horizontal_fov_deg_max)
        vertical_min = np.deg2rad(self.cfg.lidar.vertical_fov_deg_min)
        vertical_max = np.deg2rad(self.cfg.lidar.vertical_fov_deg_max)
        h_den = max(self.cfg.lidar.horizontal_line_num - 1, 1)
        v_den = max(self.cfg.lidar.vertical_line_num - 1, 1)

        ray_vectors = torch.zeros(
            self.cfg.lidar.vertical_line_num,
            self.cfg.lidar.horizontal_line_num,
            3,
            dtype=torch.float,
            device=self.device,
        )
        for i in range(self.cfg.lidar.vertical_line_num):
            elevation = vertical_max - (vertical_max - vertical_min) * (i / v_den)
            for j in range(self.cfg.lidar.horizontal_line_num):
                azimuth = horizontal_max - (horizontal_max - horizontal_min) * (j / h_den)
                ray_vectors[i, j, 0] = np.cos(azimuth) * np.cos(elevation)
                ray_vectors[i, j, 1] = np.sin(azimuth) * np.cos(elevation)
                ray_vectors[i, j, 2] = np.sin(elevation)

        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        return ray_vectors.reshape(-1, 3)

    def _update_lidar_pose(self):
        if getattr(self, "lidar_zero_obs", False):
            return
        offset = self.lidar_position_offset.unsqueeze(0).repeat(self.num_envs, 1)
        sensor_pos = self.root_states[:, :3] + quat_apply(self.base_quat, offset)
        sensor_quat = quat_mul(self.base_quat, self.lidar_quat_offset)
        self.lidar_sensor_pos_tensor[:, 0, :] = sensor_pos
        self.lidar_sensor_quat_tensor[:, 0, :] = sensor_quat

    def _update_lidar(self):
        if getattr(self, "lidar_plane_obs", False):
            self._update_lidar_on_plane()
            return
        self._update_lidar_pose()
        lidar_points, lidar_dist_tensor = self.lidar_sensor.update()
        self.lidar_points = lidar_points
        lidar_distances = lidar_dist_tensor[:, 0].reshape(self.num_envs, -1)
        lidar_distances = torch.where(
            lidar_distances >= self.cfg.lidar.max_range,
            torch.full_like(lidar_distances, self.cfg.lidar.max_range),
            lidar_distances,
        )
        self.lidar_distances = torch.clip(
            lidar_distances,
            self.cfg.lidar.min_range,
            self.cfg.lidar.max_range,
        )

    def _update_lidar_on_plane(self):
        if self.lidar_plane_mode == "zero":
            self.lidar_distances.zero_()
            self.lidar_points.zero_()
            return

        self._update_lidar_pose()
        distances = torch.full_like(self.lidar_distances, self.cfg.lidar.max_range)
        points = torch.zeros_like(self.lidar_points)

        if self.lidar_plane_mode == "raycast":
            ray_dirs = self.lidar_ray_vectors.unsqueeze(0).repeat(self.num_envs, 1, 1)
            sensor_quat = self.lidar_sensor_quat_tensor[:, 0].unsqueeze(1).repeat(1, self.num_lidar_observations, 1)
            ray_dirs_world = quat_apply(
                sensor_quat.reshape(-1, 4),
                ray_dirs.reshape(-1, 3),
            ).reshape(self.num_envs, self.num_lidar_observations, 3)

            denom = ray_dirs_world[..., 2]
            raw_distances = -self.lidar_sensor_pos_tensor[:, 0, 2].unsqueeze(1) / torch.clamp(denom, max=-1e-6)
            hit = (
                (denom < -1e-6)
                & (raw_distances >= self.cfg.lidar.min_range)
                & (raw_distances <= self.cfg.lidar.max_range)
            )
            distances = torch.where(hit, raw_distances, distances)

            if self.cfg.lidar.pointcloud_in_world_frame:
                hit_points = self.lidar_sensor_pos_tensor[:, 0].unsqueeze(1) + distances.unsqueeze(-1) * ray_dirs_world
            else:
                hit_points = distances.unsqueeze(-1) * ray_dirs
            hit_grid = hit.reshape(
                self.num_envs,
                self.cfg.lidar.vertical_line_num,
                self.cfg.lidar.horizontal_line_num,
                1,
            )
            points[:, 0] = torch.where(
                hit_grid,
                hit_points.reshape(self.num_envs, self.cfg.lidar.vertical_line_num, self.cfg.lidar.horizontal_line_num, 3),
                points[:, 0],
            )

        self.lidar_distances = torch.clip(
            distances,
            self.cfg.lidar.min_range,
            self.cfg.lidar.max_range,
        )
        self.lidar_points = points

    def _get_lidar_observations(self):
        # Raw-distance LiDAR observation. On plane terrain, plane_mode controls whether
        # distances are analytic ray-plane hits, max-range no-hits, or legacy zeros.
        return self.lidar_distances * self.obs_scales.lidar

    def _light_vec(self, name, default):
        values = getattr(self.cfg.lighting, name, default)
        return gymapi.Vec3(values[0], values[1], values[2])

    def _setup_enhanced_lighting(self):
        if not getattr(self.cfg.lighting, "enhanced_lighting", True):
            return

        self.gym.set_light_parameters(
            self.sim,
            1,
            self._light_vec("key_light_color", [0.35, 0.35, 0.35]),
            self._light_vec("key_light_ambient", [0.08, 0.08, 0.08]),
            self._light_vec("key_light_direction", [1.0, 1.0, -1.0]),
        )
        self.gym.set_light_parameters(
            self.sim,
            2,
            self._light_vec("side_light_color", [0.12, 0.12, 0.12]),
            self._light_vec("side_light_ambient", [0.02, 0.02, 0.02]),
            self._light_vec("side_light_direction", [-1.0, 1.0, -0.5]),
        )
        self.gym.set_light_parameters(
            self.sim,
            3,
            self._light_vec("fill_light_color", [0.12, 0.12, 0.12]),
            self._light_vec("fill_light_ambient", [0.02, 0.02, 0.02]),
            self._light_vec("fill_light_direction", [1.0, -1.0, -0.5]),
        )

    def _get_lidar_points_world(self, env_id):
        if getattr(self, "lidar_zero_obs", False):
            return torch.zeros(0, 3, device=self.device)
        points = self.lidar_points[env_id, 0].reshape(-1, 3)
        distances = self.lidar_distances[env_id].reshape(-1)
        points = points[distances < self.cfg.lidar.max_range]

        if points.numel() == 0:
            return points

        max_points = int(getattr(self.cfg.lidar, "debug_max_points", points.shape[0]))
        if points.shape[0] > max_points:
            stride = max(1, points.shape[0] // max_points)
            points = points[::stride][:max_points]

        if self.cfg.lidar.pointcloud_in_world_frame:
            return points

        sensor_pos = self.lidar_sensor_pos_tensor[env_id, 0].unsqueeze(0).repeat(points.shape[0], 1)
        sensor_quat = self.lidar_sensor_quat_tensor[env_id, 0].unsqueeze(0).repeat(points.shape[0], 1)
        return sensor_pos + quat_apply(sensor_quat, points)

    def _draw_lidar_vis(self):
        env_ids = getattr(self.cfg.lidar, "debug_env_ids", [0])
        if isinstance(env_ids, int):
            env_ids = [env_ids]

        should_print = (
            getattr(self.cfg.lidar, "debug_print", False)
            and self.common_step_counter % max(1, int(getattr(self.cfg.lidar, "debug_print_interval", 50))) == 0
        )
        color = tuple(getattr(self.cfg.lidar, "debug_point_color", (0.0, 1.0, 0.2)))
        point_size = float(getattr(self.cfg.lidar, "debug_point_size", 0.015))
        sphere_geom = gymutil.WireframeSphereGeometry(point_size, 4, 4, None, color=color)

        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue
            if should_print:
                self._print_lidar_feedback(env_id)
            points = self._get_lidar_points_world(env_id).detach().cpu().numpy()
            for point in points:
                sphere_pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

    def _print_lidar_feedback(self, env_id):
        if getattr(self, "lidar_zero_obs", False):
            print("\n[lidar feedback]")
            print(f"step={self.common_step_counter} env={env_id} zero_on_plane=True")
            print(f"distances: all zeros, shape=({self.num_lidar_observations},)")
            return

        distances = self.lidar_distances[env_id].reshape(-1)
        valid = distances < self.cfg.lidar.max_range
        valid_distances = distances[valid]
        num_valid = int(valid.sum().item())
        num_total = distances.numel()

        print("\n[lidar feedback]")
        print(f"step={self.common_step_counter} env={env_id} valid_hits={num_valid}/{num_total}")
        if num_valid == 0:
            print("distances: no valid hits")
            return

        print(
            "distance stats: "
            f"min={valid_distances.min().item():.3f} "
            f"mean={valid_distances.mean().item():.3f} "
            f"max={valid_distances.max().item():.3f}"
        )

        num_points = min(int(getattr(self.cfg.lidar, "debug_print_num_points", 10)), num_valid)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()[:num_points]
        sample_distances = distances[valid_indices].detach().cpu().numpy()
        sample_points = self._get_lidar_points_world(env_id)[:num_points].detach().cpu().numpy()
        print(f"first {num_points} distances:", np.array2string(sample_distances, precision=3))
        print(f"first {sample_points.shape[0]} world points:")
        print(np.array2string(sample_points, precision=3))
    

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
        IMPORTANT: This method takes a lot of GPU memory.
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points),
                                    self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()  # convert float to indices
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _sample_terrain_heights_at_xy(self, points_xy):
        """Sample terrain heights at world-frame xy points.

        Uses the local maximum around the query cell so step edges are not
        underestimated for foot clearance.
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(points_xy.shape[:-1], device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't sample height with terrain mesh type 'none'")

        points = points_xy + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[..., 0].reshape(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[..., 1].reshape(-1), 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights4 = self.height_samples[px + 1, py + 1]
        heights = torch.maximum(torch.maximum(heights1, heights2), torch.maximum(heights3, heights4))
        return heights.view(points_xy.shape[:-1]) * self.terrain.cfg.vertical_scale
    


    def _update_feet_contact_timers(self):
        """Update foot contact timers once per physics step for gait rewards."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)

        air_time_next = self.feet_air_time + self.dt
        contact_time_next = self.feet_contact_time + self.dt
        self.feet_first_contact[:] = (self.feet_air_time > 0.) & contact_filt
        self.feet_air_time_on_contact[:] = air_time_next

        self.feet_air_time[:] = air_time_next * ~contact_filt
        self.feet_contact_time[:] = contact_time_next * contact_filt
        self.last_contacts[:] = contact

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """

        self.gym.clear_lines(self.viewer)

        if getattr(self.cfg.lidar, "enable", False) and getattr(self.cfg.lidar, "debug_viz", False):
            self._draw_lidar_vis()

        # draw height lines
        if not self.debug_viz:
            return
        if not ("terrain" in self.__dir__() and self.terrain.cfg.measure_heights):
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_acc[:] = self.base_lin_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_acc[:] = self.base_ang_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13] - self.last_root_vel[:, 3:]) / self.dt
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.episode_base_height_sum += self.base_pos[:, 2]
        self.episode_base_height_count += 1.
        self._update_feet_contact_timers()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and (
            self.debug_viz or getattr(self.cfg.lidar, "debug_viz", False)
        ):
            self._draw_debug_vis()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.embedded_state_transition_time[env_ids] = 0.0
        self.embedded_state_default_dof_pos[env_ids] = self._get_embedded_state_target_dof_pos()[env_ids]
        self.filtered_embedded_state_default_dof_pos[env_ids] = self.default_dof_pos


    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, 0:1] * self.command_lin_vel_scale,
                                  self.commands[:, 1:2] * self.command_lin_vel_scale,
                                  self.commands[:, 2:3] * self.command_ang_vel_scale,
                                  self.commands[:, 4:5] * self.command_embedded_state_scale,
                                  (self.dof_pos - self.embedded_state_default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  ), dim=-1)
        # add perceptive inputs if not blind
        if getattr(self.cfg.lidar, "enable", False):
            self.obs_buf = torch.cat((self.obs_buf, self._get_lidar_observations()), dim=-1)
        elif self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.45 -
                                 self.measured_heights, -5, 5.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # print("commands:",self.commands[:5,:5])

    def _get_embedded_state_target_dof_pos(self):
        embedded_state = self.commands[:, 4].unsqueeze(1)
        return self.default_dof_pos + embedded_state * (self.mammal_default_dof_pos - self.default_dof_pos)

    def _update_embedded_state_transition_time(self):
        self.embedded_state_transition_time += self.dt

    def _get_embedded_state_reward_ramp(self):
        ramp_time = max(float(getattr(self.cfg.rewards, "structure_transition_reward_ramp_time", 1.0)), 1e-6)
        ramp = torch.clamp(self.embedded_state_transition_time / ramp_time, 0.0, 1.0)
        return torch.where(self.embedded_state_transition_time <= ramp_time, ramp, torch.zeros_like(ramp))

    def _get_gait_reward_active_mask(self):
        ramp_time = max(float(getattr(self.cfg.rewards, "structure_transition_reward_ramp_time", 1.0)), 1e-6)
        return (self.embedded_state_transition_time > ramp_time).float()
        

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)
                   == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._update_embedded_state_transition_time()
        self.embedded_state_default_dof_pos = self._get_embedded_state_target_dof_pos()
        if self.cfg.commands.heading_command:
            # Convert heading to ang vel command useing P controller
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if getattr(self.cfg.lidar, "enable", False):
            self._update_lidar()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        embedded_state_one_prob = float(self.command_ranges.get("embedded_state_one_prob", 0.5))
        embedded_state_one_prob = min(max(embedded_state_one_prob, 0.0), 1.0)
        self.commands[env_ids, 4] = (
            torch.rand(len(env_ids), device=self.device) < embedded_state_one_prob
        ).float()


        # self.commands[env_ids,4] = 1.0
        # print(self.commands[env_ids,4])

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type  == "P_LOWPASS":
            tau = max(float(getattr(self.cfg.control, "default_dof_pos_filter_tau", 0.3)), 1e-6)
            alpha = 1.0 - np.exp(-self.sim_params.dt / tau)
            self.filtered_embedded_state_default_dof_pos += alpha * (
                self.embedded_state_default_dof_pos - self.filtered_embedded_state_default_dof_pos
            )
            done_threshold = float(getattr(self.cfg.control, "default_dof_pos_filter_done_threshold", 0.02))
            
            filter_error = torch.max(
                torch.abs(self.filtered_embedded_state_default_dof_pos - self.embedded_state_default_dof_pos),
                dim=1,
            ).values
            use_final_default = (filter_error < done_threshold).unsqueeze(1)
            default_dof_pos = torch.where(
                use_final_default,
                self.embedded_state_default_dof_pos,
                self.filtered_embedded_state_default_dof_pos,
            )
            torques = self.p_gains*(actions_scaled + default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type == "V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains * \
                (self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * \
            torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        move_up_ratio = getattr(self.cfg.terrain, "terrain_curriculum_move_up_distance_ratio", 0.5)
        move_down_ratio = getattr(self.cfg.terrain, "terrain_curriculum_move_down_command_distance_ratio", 0.5)
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length * move_up_ratio
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * move_down_ratio) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        tracking_threshold = getattr(self.cfg.commands, "tracking_lin_vel_curriculum_threshold", 0.8)
        curriculum_step = getattr(self.cfg.commands, "command_curriculum_step", 0.5)
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > tracking_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - curriculum_step, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + curriculum_step, 0., self.cfg.commands.max_curriculum)


    
    def _debug_info(self):
        """Print debug information for the specified environment(s)"""
        if not self.cfg.env.debug_mode:
            return
        
        self.debug_step_counter += 1
        if self.debug_step_counter % self.cfg.env.debug_interval != 0:
            return
        
        env_id = self.cfg.env.debug_env_id
        
        # Determine which environments to print
        if env_id == -1:
            # Print all environments
            env_ids_to_print = list(range(self.num_envs))
        else:
            # Print specified environment
            if env_id >= self.num_envs:
                return
            env_ids_to_print = [env_id]
        
        print("\n" + "="*80)
        print(f"DEBUG INFO - Step {self.common_step_counter} | Printing {len(env_ids_to_print)} environment(s)")
        print("="*80)
        
        for env_idx in env_ids_to_print:
            print(f"\n{'─'*80}")
            print(f"Environment {env_idx} | Episode Length: {self.episode_length_buf[env_idx].item()}")
            print(f"{'─'*80}")
            
            # Base state
            print(f"\n[Base State]")
            print(f"  Position:     [{self.base_pos[env_idx, 0]:.3f}, {self.base_pos[env_idx, 1]:.3f}, {self.base_pos[env_idx, 2]:.3f}]")
            print(f"  Base Height:  {self.base_pos[env_idx, 2]:.3f} m")

            # Contact info and forces
            contact = self.contact_forces[env_idx, self.feet_indices, 2] > 1.
            contact_forces = self.contact_forces[env_idx, self.feet_indices, :]
            contact_forces_z = contact_forces[:, 2]
            max_contact_force = torch.max(contact_forces_z).item()
            max_force_idx = torch.argmax(contact_forces_z).item()
            
            # Foot names for better readability
            foot_names = getattr(self, "feet_names", ['LB_FOOT', 'LF_FOOT', 'LM_FOOT', 'RB_FOOT', 'RF_FOOT', 'RM_FOOT'])
            short_foot_names = [name.replace("_FOOT", "") for name in foot_names]
            
            print(f"\n[Contact Info]")
            print(f"  Feet Contact: {contact.cpu().numpy()}")
            print(f"  Feet Order: {short_foot_names}")
            group1_contact = contact[self.tripod_group1_indices].cpu().numpy()
            group2_contact = contact[self.tripod_group2_indices].cpu().numpy()
            group1_names = [short_foot_names[i] for i in self.tripod_group1_indices.cpu().tolist()]
            group2_names = [short_foot_names[i] for i in self.tripod_group2_indices.cpu().tolist()]
            print(f"  Tripod Group 1 {group1_names}: {group1_contact}")
            print(f"  Tripod Group 2 {group2_names}: {group2_contact}")
            print(f"  Contact Forces (X,Y,Z) [N]:")
            for i, name in enumerate(short_foot_names):
                fx = contact_forces[i, 0].item()
                fy = contact_forces[i, 1].item()
                fz = contact_forces[i, 2].item()
                f_total = torch.norm(contact_forces[i]).item()
                contact_str = "✓" if contact[i].item() else "✗"
                print(f"    {name}: [{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] (Total: {f_total:7.2f}) {contact_str}")
            print(f"  Max Contact Force: {max_contact_force:.2f} N (Foot {short_foot_names[max_force_idx]})")
            print(f"  Total Ground Force: {torch.sum(contact_forces_z).item():.2f} N")
            print(f"  Feet Air Time: {self.feet_air_time[env_idx].cpu().numpy()}")
            
        
        print("\n" + "="*80 + "\n")





#-----------------reward functions---------------------#
#-----------------reward functions---------------------#
#-----------------reward functions---------------------#
    
    def _reward_feet_air_time(self):
        # Reward long steps
        air_time_target = getattr(self.cfg.rewards, "feet_air_time_target", 0.15)
        rew_airTime = torch.sum((self.feet_air_time_on_contact - air_time_target) * self.feet_first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        return rew_airTime

    def _reward_tripod_contact_pattern(self):
        """Penalize contact patterns that are not two opposite tripod groups."""
        contact_threshold = getattr(self.cfg.rewards, "tripod_contact_threshold", 1.0)
        min_command = getattr(self.cfg.rewards, "tripod_contact_min_command", 0.1)

        contact = (self.contact_forces[:, self.feet_indices, 2] > contact_threshold).float()
        group1 = contact[:, self.tripod_group1_indices]
        group2 = contact[:, self.tripod_group2_indices]

        # Feet in the same tripod should share the same contact state.
        group1_same = torch.var(group1, dim=1, unbiased=False)
        group2_same = torch.var(group2, dim=1, unbiased=False)

        # The two tripod groups should be opposite: one stance, one swing.
        group1_mean = torch.mean(group1, dim=1)
        group2_mean = torch.mean(group2, dim=1)
        group_opposite = torch.square(group1_mean + group2_mean - 1.0)

        moving = (torch.norm(self.commands[:, :2], dim=1) > min_command).float()
        return (group1_same + group2_same + group_opposite) * moving * self._get_gait_reward_active_mask()

    def _reward_embedded_state_dof_pos(self):
        return torch.mean(torch.square(self.dof_pos - self.embedded_state_default_dof_pos), dim=1) * self._get_embedded_state_reward_ramp()

    def _reward_embedded_state_dof_vel(self):
        pos_error = torch.mean(torch.abs(self.dof_pos - self.embedded_state_default_dof_pos), dim=1)
        transition_mask = (
            pos_error > getattr(self.cfg.rewards, "structure_transition_error_threshold", 0.05)
        ).float()
        return torch.mean(torch.square(self.dof_vel), dim=1) * transition_mask * self._get_embedded_state_reward_ramp()

    def _reward_mammal_haa_range(self):
        """Soft-limit HAA spread in mammal mode without fixing it to one pose."""
        mammal_state_mask = (self.commands[:, 4] >= 0.5).float()
        target = self.mammal_default_dof_pos[:, self.haa_indices]
        range_limit = getattr(self.cfg.rewards, "mammal_haa_range_limit", 0.25)
        range_error = torch.abs(self.dof_pos[:, self.haa_indices] - target) - range_limit
        return (
            torch.mean(torch.square(torch.clamp(range_error, min=0.0)), dim=1)
            * mammal_state_mask
            * self._get_gait_reward_active_mask()
        )

    def _reward_mammal_thigh_motion(self):
        """Discourage walking by swinging HAA/HFE, but allow HAA for yaw turning."""
        mammal_state_mask = (self.commands[:, 4] >= 0.5).float()
        min_command = getattr(self.cfg.rewards, "mammal_thigh_motion_min_command", 0.1)
        moving = (torch.norm(self.commands[:, :2], dim=1) > min_command).float()
        turn_relief_command = max(float(getattr(self.cfg.rewards, "mammal_haa_turn_relief_command", 0.25)), 1e-6)
        yaw_command = torch.abs(self.commands[:, 2])
        haa_weight = 1.0 - torch.clamp(yaw_command / turn_relief_command, 0.0, 1.0)

        haa_vel = torch.mean(torch.square(self.dof_vel[:, self.haa_indices]), dim=1)
        hfe_vel = torch.mean(torch.square(self.dof_vel[:, self.hfe_indices]), dim=1)
        haa_action_delta = torch.mean(
            torch.square(self.actions[:, self.haa_indices] - self.last_actions[:, self.haa_indices]),
            dim=1,
        )
        hfe_action_delta = torch.mean(
            torch.square(self.actions[:, self.hfe_indices] - self.last_actions[:, self.hfe_indices]),
            dim=1,
        )
        return (
            haa_weight * (haa_vel + haa_action_delta)
            + hfe_vel
            + hfe_action_delta
        ) * mammal_state_mask * moving * self._get_gait_reward_active_mask()

    def _sync_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Penalize desynchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        return se_air + se_contact
    
    def _async_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Penalize synchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        return se_act_0 + se_act_1

    def _reward_feet_sync(self):
        """
        Penalize desynchronization within each tripod group by summing all pair-wise sync errors.
        Group 1: LF (1), LB (0), RM (5)
        Group 2: RF (4), RB (3), LM (2)
        """
        # Pairs in Group 1
        sync_g1 = self._sync_reward_func(1, 0) + self._sync_reward_func(1, 5) + self._sync_reward_func(0, 5)
        
        # Pairs in Group 2
        sync_g2 = self._sync_reward_func(4, 3) + self._sync_reward_func(4, 2) + self._sync_reward_func(3, 2)

        # Total sync penalty is the sum of penalties from both groups
        sync_reward = sync_g1 + sync_g2
        # Only apply reward when moving
        if self.cfg.commands.heading_command:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 3]) > 0.1)
        else:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 2]) > 0.1)
        
        return sync_reward * move_condition * self._get_gait_reward_active_mask()

    def _reward_feet_async(self):
        async_reward = 0
        # Sum of async penalties for all pairs between Group 1 and Group 2
        for foot_g1 in self.tripod_group1_indices:
            for foot_g2 in self.tripod_group2_indices:
                async_reward += self._async_reward_func(foot_g1, foot_g2)

        # Only apply reward when moving
        if self.cfg.commands.heading_command:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 3]) > 0.1)
        else:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 2]) > 0.1)

        return async_reward * move_condition * self._get_gait_reward_active_mask()
    
    def _reward_contact_force_balance(self):
        """
        核心惩罚逻辑：惩罚接触力方差。
        惩罚同一三角支撑组内接触力的方差。
        这会鼓励处于同一支撑相的腿均匀地分担负载。
        """
        # 获取所有脚底的法向接触力（Z轴方向）
        normal_forces = self.contact_forces[:, self.feet_indices, 2]
        
        # 创建一个蒙版，标记哪些脚正在与地面接触（力大于1.0）
        contact_mask = (normal_forces > 1.0).float()

        # --- 处理第一组腿 ---
        # 获取第一组腿的接触力
        forces_g1 = normal_forces[:, self.tripod_group1_indices]
        # 获取第一组腿的接触蒙版
        mask_g1 = contact_mask[:, self.tripod_group1_indices]
        # 将未接触地面的腿的力置零，以便它们不影响均值和方差的计算
        masked_forces_g1 = forces_g1 * mask_g1
        # 计算第一组中接触地面的腿的数量
        num_contacting_g1 = torch.sum(mask_g1, dim=1)
        # 只有当接触地面的腿数大于1时，计算方差才有意义
        is_valid_g1 = (num_contacting_g1 > 1).float()
        # 计算接触腿的平均力（分母加一个小数防止除以零）
        mean_g1 = torch.sum(masked_forces_g1, dim=1) / (num_contacting_g1 + 1e-6)
        # 计算接触腿的力的方差： Variance = mean( (x - mean)^2 )
        variance_g1 = torch.sum(torch.square(masked_forces_g1 - mean_g1.unsqueeze(1)) * mask_g1, dim=1) / (num_contacting_g1 + 1e-6)
        
        # --- 处理第二组腿 ---
        # 获取第二组腿的接触力
        forces_g2 = normal_forces[:, self.tripod_group2_indices]
        # 获取第二组腿的接触蒙版
        mask_g2 = contact_mask[:, self.tripod_group2_indices]
        # 将未接触地面的腿的力置零
        masked_forces_g2 = forces_g2 * mask_g2
        # 计算第二组中接触地面的腿的数量
        num_contacting_g2 = torch.sum(mask_g2, dim=1)
        # 只有当接触地面的腿数大于1时，计算方差才有意义
        is_valid_g2 = (num_contacting_g2 > 1).float()
        # 计算接触腿的平均力
        mean_g2 = torch.sum(masked_forces_g2, dim=1) / (num_contacting_g2 + 1e-6)
        # 计算接触腿的力的方差
        variance_g2 = torch.sum(torch.square(masked_forces_g2 - mean_g2.unsqueeze(1)) * mask_g2, dim=1) / (num_contacting_g2 + 1e-6)

        # 总惩罚是两组方差的总和，仅在对应组有效（接触腿数>1）时计算
        total_variance_penalty = variance_g1 * is_valid_g1 + variance_g2 * is_valid_g2
        
        return total_variance_penalty

    def _reward_shank_vertical(self):

         # 获取小腿的刚体状态: [num_envs, num_shanks, 13]
        # 13维度: pos(3), quat(4), lin_vel(3), ang_vel(3)
        spider_state_mask = (self.commands[:, 4] < 0.5).float()
        shank_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.shank_indices, :]
        foot_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :]

        x_error = shank_states[:, :, 0] - foot_states[:, :, 0]
        y_error = shank_states[:, :, 1] - foot_states[:, :, 1]
        
        # 计算小腿方向向量在XY平面的投影长度    
        horizontal_dist = torch.sum(torch.sqrt(x_error**2 + y_error**2), dim=1)

        return horizontal_dist * spider_state_mask


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        embedded_state = self.commands[:, 4]
        spider_target = getattr(self.cfg.rewards, "base_height_spider_target", getattr(self.cfg.rewards, "base_height_target", 0.45))
        mammal_target = getattr(self.cfg.rewards, "base_height_mammal_target", spider_target)
        base_height_target = spider_target + embedded_state * (mammal_target - spider_target)
        return torch.square(base_height - base_height_target) * self._get_gait_reward_active_mask()
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.embedded_state_default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < self.speed_min)

    

# Bodies:  ['base_link', 'LB_HIP', 'LB_THIGH', 'LB_SHANK', 'LB_FOOT', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 
# 'LM_HIP', 'LM_THIGH', 'LM_SHANK', 'LM_FOOT', 'RB_HIP', 'RB_THIGH', 'RB_SHANK', 'RB_FOOT', 
# 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RM_HIP', 'RM_THIGH', 'RM_SHANK', 'RM_FOOT']
