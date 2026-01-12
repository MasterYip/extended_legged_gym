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

import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict

from legged_gym.envs.batch_rollout.robot_traj_grad_sampling import RobotTrajGradSampling
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.batch_rollout.go2_traj_grad_sampling_config import Go2TrajGradSamplingCfg


class Go2TrajGradSampling(RobotTrajGradSampling):
    """
    Go2 robot environment with trajectory gradient sampling for optimization.

    This environment extends the RobotTrajGradSampling capabilities with Go2-specific
    configurations and implementations.
    """

    cfg: Go2TrajGradSamplingCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """
        Initialize the Go2 trajectory gradient sampling environment.

        Args:
            cfg: Environment configuration
            sim_params: Simulation parameters
            physics_engine: Physics engine type
            sim_device: Simulation device
            headless: Whether to run in headless mode
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Initialize gait parameters similar to unitree_go2_env.py
        self._gait = "trot"  # Default gait
        self._gait_phase = {
            "stand": torch.zeros(4, device=self.device),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25], device=self.device),
            "trot": torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device),
            "canter": torch.tensor([0.0, 0.33, 0.33, 0.66], device=self.device),
            "gallop": torch.tensor([0.0, 0.05, 0.4, 0.35], device=self.device),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": torch.tensor([1.0, 1.0, 0.0], device=self.device),
            "walk": torch.tensor([0.75, 1.0, 0.08], device=self.device),
            "trot": torch.tensor([0.45, 2.0, 0.08], device=self.device),
            "canter": torch.tensor([0.4, 4.0, 0.06], device=self.device),
            "gallop": torch.tensor([0.3, 3.5, 0.10], device=self.device),
        }

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
        Must be adapted for AnymalC's observation structure.

        Args:
            cfg: Environment config file

        Returns:
            Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # Fill in noise scales for appropriate observation ranges
        # First 3: linear velocity
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # Next 3: angular velocity
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # Next 3: gravity direction
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # Next 4: commands (no noise)
        noise_vec[9:13] = 0.
        # Joint positions and velocities (adjust indices based on AnymalC's DOFs)
        noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # Previous actions (no noise)
        noise_vec[37:49] = 0.

        # Add noise to height measurements if used
        if self.cfg.terrain.measure_heights:
            # Adjust the index based on AnymalC's observation structure
            noise_vec[49:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def get_foot_step(self, duty_ratio, cadence, amplitude, phases, time):
        """Compute target foot heights based on gait parameters.

        Args:
            duty_ratio: Ratio of stance phase to full gait cycle
            cadence: Frequency of the gait cycle
            amplitude: Maximum height of foot during swing phase
            phases: Phase offset for each foot [0-1]
            time: Current time in seconds

        Returns:
            Tensor of shape [num_feet] with target foot heights
        """
        # Calculate the normalized phase for each foot
        freq = cadence
        gait_phase = torch.fmod(time * freq + phases, 1.0)

        # Compute the foot height based on the phase
        # When in stance phase (determined by duty_ratio), height is 0
        # When in swing phase, height follows a sine curve scaled by amplitude
        stance_phase = gait_phase < duty_ratio
        swing_phase = ~stance_phase
        swing_phase_normalized = (gait_phase[swing_phase] - duty_ratio) / (1.0 - duty_ratio)

        # Initialize heights to zero
        heights = torch.zeros_like(gait_phase)

        # Apply sine curve for swing phase
        heights[swing_phase] = amplitude * torch.sin(swing_phase_normalized * torch.pi)

        return heights

    # === Reward Functions from unitree_go2_env.py ===

    def _reward_gaits(self):
        """Reward for tracking target foot height based on gait pattern."""
        # Get current foot heights
        z_feet = self.foot_positions[:, :, 2]

        # Get parameters for the current gait
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]

        # Calculate target foot heights based on gait pattern
        # Create a time tensor for each environment
        time = torch.ones_like(self.commands[:, 0]) * self.t_main

        # Calculate target heights for each environment
        # NOTE: main env reward is not used, ignore it
        z_feet_tar = self.get_foot_step(duty_ratio, cadence, amplitude, phases, self.t_rollout)
        z_feet_tar = z_feet_tar.unsqueeze(0).repeat(self.total_num_envs, 1)

        # Compute squared error normalized by tolerance
        error = ((z_feet_tar - z_feet) / 0.05) ** 2
        return -torch.sum(error, dim=1)

    def _reward_air_time(self):
        """Reward for appropriate foot air time, similar to feet_air_time but with
        specific modifications for AnymalC's locomotion pattern."""
        # Check foot contacts (for all feet)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        # Filter contacts
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # Check first contact after being in the air
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt

        # Calculate reward based on air time
        rew_airTime = torch.sum((self.feet_air_time - 0.1) * first_contact, dim=1)

        # Update feet air time
        self.feet_air_time *= ~contact_filt

        return rew_airTime

    def _reward_pos(self):
        """Reward for tracking target position."""
        # Get current position (base position)
        pos = self.root_states[:, :3]

        # Create a target position based on velocity commands
        # Use time elapsed instead of episode_length_buf
        time_elapsed = torch.ones_like(self.commands[:, 0]) * self.t_main
        pos_tar = self.commands[:, :3] * self.dt * time_elapsed.unsqueeze(1)

        # For more accurate head position, use direction vector transform
        head_vec = torch.tensor([0.285, 0.0, 0.0], device=self.device)
        forward = quat_apply(self.base_quat, head_vec.repeat(pos.shape[0], 1))
        head_pos = pos + forward * 0.285  # Using the forward vector length as scaling

        # Compute position error
        pos_error = torch.sum((head_pos - pos_tar) ** 2, dim=1)

        return -pos_error

    def _reward_upright(self):
        """Reward for maintaining upright orientation."""
        # Use projected gravity instead of direct quaternion conversion
        # Penalize deviation from the upright position
        up_vec = torch.zeros_like(self.projected_gravity)
        up_vec[:, 2] = -1.0  # Upright is when gravity points down in body frame

        # Compute orientation error
        orientation_error = torch.sum((self.projected_gravity - up_vec)**2, dim=1)

        return -orientation_error

    def _reward_yaw(self):
        # FIXME: not aligned with dial-mpc
        """Reward for tracking target yaw orientation."""
        # Instead of euler angles, use forward vector projection to get yaw
        forward = quat_apply_yaw(self.base_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.total_num_envs, 1))
        current_yaw = torch.atan2(forward[:, 1], forward[:, 0])

        # Target yaw based on commands
        # For simplicity, just use the command as the target without accumulation
        target_yaw = self.commands[:, 3] if self.cfg.commands.heading_command else 0.0

        # Compute yaw difference (handling circular difference)
        yaw_diff = current_yaw - target_yaw
        # Normalize to [-π, π]
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        yaw_error = torch.square(yaw_diff)

        return -yaw_error

    def _reward_vel(self):
        """Reward for tracking target linear velocity."""
        # Linear velocity is already in body frame
        body_vel = self.base_lin_vel

        # Compute velocity tracking error for xy components
        vel_error = torch.sum((body_vel[:, :2] - self.commands[:, :2])**2, dim=1)

        return -vel_error

    def _reward_ang_vel(self):
        """Reward for tracking target angular velocity."""
        # Angular velocity is already in body frame
        body_ang_vel = self.base_ang_vel

        # Compute angular velocity tracking error for yaw component
        ang_vel_error = torch.square(body_ang_vel[:, 2] - self.commands[:, 2])

        return -ang_vel_error

    def _reward_height(self):
        """Reward for maintaining target body height."""
        # Current base height
        current_height = self.root_states[:, 2]

        # Target height
        target_height = self.cfg.rewards.base_height_target

        # Compute height error
        height_error = torch.square(current_height - target_height)

        return -height_error

    def _reward_energy(self):
        """Penalty for excessive energy consumption (high torque × velocity)."""
        # Power = torque × angular velocity
        power = torch.clamp(self.torques * self.dof_vel, min=0.0)

        # Normalize and square to penalize high power consumption
        normalized_power = power / 160.0  # Normalization factor
        energy_penalty = -torch.sum(normalized_power**2, dim=1)

        return energy_penalty

    def _reward_alive(self):
        """Reward for staying alive (not terminating)."""
        return 1.0 - self.reset_buf.long()

    def _draw_debug_vis(self):
        """Draw debug visualizations for AnymalC.
        Extends the base class debug visualization with robot-specific elements.
        """
        # Call parent class visualization
        super()._draw_debug_vis()

        # Add AnymalC-specific visualizations if viewer is available
        if hasattr(self, 'vis') and self.vis is not None:
            # Visualize linear velocity and command velocity
            lin_vel = self.root_states[:, 7:10].cpu().numpy()
            cmd_vel_world = quat_apply_yaw(self.base_quat, self.commands[:, :3]).cpu().numpy()
            cmd_vel_world[:, 2] = 0.0

            # Draw arrows for each main environment
            for j in range(len(self.main_env_indices)):
                i = self.main_env_indices[j]
                base_pos = self.root_states[i, :3].cpu().numpy()
                # Draw current velocity (green)
                self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
                # Draw command velocity (red)
                self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))

            # Draw foot trajectories if available
            if hasattr(self, 'foot_positions'):
                for j in range(len(self.main_env_indices)):
                    i = self.main_env_indices[j]
                    # Get foot positions for this environment
                    for k in range(4):  # 4 feet for AnymalC
                        foot_pos = self.foot_positions[i, k].cpu().numpy()
                        # Draw spheres at foot positions
                        foot_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
                        self.vis.draw_point(i, foot_pos, size=0.03, color=foot_colors[k])
