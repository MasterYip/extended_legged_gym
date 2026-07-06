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

from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import El4090SpiderCfg


class EL_4090(ElSpider):
    cfg : El4090SpiderCfg
     # env Init
    def __init__(self, cfg: El4090SpiderCfg, sim_params, physics_engine, sim_device, headless,task_name="el4090_spider"):
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
    

    def compute_observations(self):
        super().compute_observations()
        # print(self.obs[0,66:187])

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

        # draw height lines
        if not ("terrain" in self.__dir__() and self.terrain.cfg.measure_heights):
            return
        self.gym.clear_lines(self.viewer)
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

        

        # self._draw_debug_vis()

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
        return (group1_same + group2_same + group_opposite) * moving

    def _reward_feet_terrain_clearance(self):
        """Penalize swing feet that do not clear nearby terrain."""
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        contact_threshold = getattr(self.cfg.rewards, "feet_clearance_contact_threshold", 1.0)
        clearance = getattr(self.cfg.rewards, "feet_clearance_target", 0.12)
        lookahead = getattr(self.cfg.rewards, "feet_clearance_lookahead", 0.15)
        min_command = getattr(self.cfg.rewards, "feet_clearance_min_command", 0.1)

        command_xy = self.commands[:, :2]
        command_norm = torch.norm(command_xy, dim=1, keepdim=True)
        moving = (command_norm.squeeze(1) > min_command).float()

        command_dir_base = command_xy / torch.clamp(command_norm, min=1e-6)
        command_dir_world = quat_apply_yaw(
            self.base_quat,
            torch.cat((command_dir_base, torch.zeros(self.num_envs, 1, device=self.device)), dim=1),
        )[:, :2]

        sample_xy = self.foot_positions[:, :, :2] + command_dir_world.unsqueeze(1) * lookahead
        terrain_heights = self._sample_terrain_heights_at_xy(sample_xy)
        target_foot_z = terrain_heights + clearance

        swing_mask = (self.contact_forces[:, self.feet_indices, 2] <= contact_threshold).float()
        clearance_error = torch.clamp(target_foot_z - self.foot_positions[:, :, 2], min=0.0)
        penalty = torch.sum(torch.square(clearance_error) * swing_mask, dim=1) * moving

        if getattr(self.cfg.rewards, "debug_feet_clearance", False):
            interval = max(1, int(getattr(self.cfg.rewards, "debug_feet_clearance_interval", 100)))
            if self.common_step_counter % interval == 0:
                env_id = int(getattr(self.cfg.rewards, "debug_feet_clearance_env_id", 0))
                env_id = max(0, min(env_id, self.num_envs - 1))
                foot_names = ["LB", "LF", "LM", "RB", "RF", "RM"]
                print("\n[feet_terrain_clearance debug]")
                print(f"step={self.common_step_counter} env={env_id}")
                print(f"command_xy={command_xy[env_id].detach().cpu().numpy()} "
                      f"command_norm={command_norm[env_id, 0].item():.4f} moving={moving[env_id].item():.0f}")
                print(f"command_dir_world={command_dir_world[env_id].detach().cpu().numpy()} "
                      f"lookahead={lookahead:.3f} clearance={clearance:.3f}")
                print(f"penalty_unscaled={penalty[env_id].item():.6f}")
                for foot_id, foot_name in enumerate(foot_names[:len(self.feet_indices)]):
                    print(
                        f"  {foot_name}: "
                        f"sample_xy={sample_xy[env_id, foot_id].detach().cpu().numpy()} "
                        f"terrain_z={terrain_heights[env_id, foot_id].item():.4f} "
                        f"foot_z={self.foot_positions[env_id, foot_id, 2].item():.4f} "
                        f"target_z={target_foot_z[env_id, foot_id].item():.4f} "
                        f"contact_fz={self.contact_forces[env_id, self.feet_indices[foot_id], 2].item():.4f} "
                        f"swing={swing_mask[env_id, foot_id].item():.0f} "
                        f"err={clearance_error[env_id, foot_id].item():.4f}"
                    )

        return penalty
 
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
        
        return sync_reward * move_condition

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

        return async_reward * move_condition
    
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
        shank_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.shank_indices, :]
        foot_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :]

        x_error = shank_states[:, :, 0] - foot_states[:, :, 0]
        y_error = shank_states[:, :, 1] - foot_states[:, :, 1]
        
        # 计算小腿方向向量在XY平面的投影长度    
        horizontal_dist = torch.sum(torch.sqrt(x_error**2 + y_error**2), dim=1)

        return horizontal_dist

    



    def _reward_stand_on_six_legs(self):
        # 低命令下：鼓励六条腿全部着地

        lin_cmd_small = torch.norm(self.commands[:, :2], dim=1) < self.speed_min
        if self.cfg.commands.heading_command:
            yaw_or_heading_small = torch.abs(self.commands[:, 3]) < self.speed_min
        else:
            yaw_or_heading_small = torch.abs(self.commands[:, 2]) < self.speed_min
        small_command_mask = torch.logical_and(lin_cmd_small, yaw_or_heading_small)

        # 足端接触（法向力阈值 1N）
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        num_feet_in_contact = torch.sum(foot_contact.float(), dim=1)

        # 惩罚未着地的脚数量；六足都着地时为 0
        missing_contact_penalty = len(self.feet_indices) - num_feet_in_contact

        return missing_contact_penalty * small_command_mask.float()

    
    

# Bodies:  ['base_link', 'LB_HIP', 'LB_THIGH', 'LB_SHANK', 'LB_FOOT', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 
# 'LM_HIP', 'LM_THIGH', 'LM_SHANK', 'LM_FOOT', 'RB_HIP', 'RB_THIGH', 'RB_SHANK', 'RB_FOOT', 
# 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RM_HIP', 'RM_THIGH', 'RM_SHANK', 'RM_FOOT']
