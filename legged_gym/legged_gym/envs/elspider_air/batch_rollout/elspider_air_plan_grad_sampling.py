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

import torch
import numpy as np
from typing import Optional

from legged_gym.envs.batch_rollout.robot_plan_grad_sampling import RobotPlanGradSampling
from legged_gym.envs.elspider_air.batch_rollout.elspider_air_plan_grad_sampling_config import ElSpiderAirPlanGradSamplingCfg
from legged_gym.utils.gait_scheduler import AsyncGaitScheduler, AsyncGaitSchedulerCfg
from legged_gym.utils.helpers import class_to_dict

class ElSpiderAirPlanGradSampling(RobotPlanGradSampling):
    """ElSpider Air robot environment with planning-based trajectory gradient sampling.

    This class extends RobotPlanGradSampling to provide ElSpider Air specific functionality
    for state velocity trajectory optimization and planning-based control.

    Key features:
    - 6-legged robot (ElSpider Air) with 18 DOF (3 joints per leg)
    - State velocity optimization (24-dim: 3 base linear + 3 base angular + 18 joint velocities)
    - Trajectory integration instead of physics simulation for rollouts
    - ElSpider Air specific rewards and observations
    """

    cfg: ElSpiderAirPlanGradSamplingCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize ElSpider Air planning environment.

        Args:
            cfg: ElSpider Air specific configuration
            sim_params: Simulation parameters
            physics_engine: Physics engine to use
            sim_device: Device for simulation
            headless: Whether to run without rendering
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # ElSpider Air specific initialization
        self.num_legs = 6
        self.num_joints_per_leg = 3

        # Verify state velocity dimension matches ElSpider Air configuration
        expected_state_vel_dim = 3 + 3 + 18  # base_lin_vel + base_ang_vel + joint_vel
        assert self.state_vel_dim == expected_state_vel_dim, \
            f"State velocity dimension mismatch: expected {expected_state_vel_dim}, got {self.state_vel_dim}"

        # ElSpider Air specific joint limits
        self._init_elspider_joint_limits()

        # Gait
        # Initialize gait parameters similar to unitree_go2_env.py
        self._gait = "trot"  # Default gait
        self._gait_phase = {
            "stand": torch.zeros(4, device=self.device),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25], device=self.device),
            "trot": torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, 0.5], device=self.device),
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
        cfg = AsyncGaitSchedulerCfg()
        self.async_gait_scheduler = AsyncGaitScheduler(self.height_samples,
                                                       self.base_quat,
                                                       self.base_lin_vel,
                                                       self.base_ang_vel,
                                                       self.projected_gravity,
                                                       self.dof_pos,
                                                       self.dof_vel,
                                                       self.foot_positions,
                                                       self.foot_velocities,
                                                       self.total_num_envs,
                                                       self.device,
                                                       cfg)

    def _init_elspider_joint_limits(self):
        """Initialize ElSpider Air specific joint limits for planning."""
        if self.enforce_joint_limits and hasattr(self, 'dof_pos_limits'):
            # ElSpider Air joint limits (in radians)
            # HAA (Hip Abduction/Adduction): ±45 degrees
            # HFE (Hip Flexion/Extension): 0 to 90 degrees
            # KFE (Knee Flexion/Extension): 0 to 90 degrees

            joint_limits = {
                'HAA': [-0.785, 0.785],    # ±45 degrees
                'HFE': [-0.2, 1.57],       # -11.5 to 90 degrees
                'KFE': [-1.57, 0.2],       # -90 to 11.5 degrees
            }

            # Apply limits to all 18 joints (6 legs × 3 joints per leg)
            for i, joint_name in enumerate(self.dof_names):
                if 'HAA' in joint_name:
                    self.dof_pos_limits[i, :] = torch.tensor(joint_limits['HAA'], device=self.device)
                elif 'HFE' in joint_name:
                    self.dof_pos_limits[i, :] = torch.tensor(joint_limits['HFE'], device=self.device)
                elif 'KFE' in joint_name:
                    self.dof_pos_limits[i, :] = torch.tensor(joint_limits['KFE'], device=self.device)

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

    def _update_derived_states(self, env_indices: torch.Tensor):
        """Override to add ElSpider Air specific state updates."""
        # Call parent method for basic state updates
        super()._update_derived_states(env_indices)

        # ElSpider Air specific foot positions are already available from rigid_body_state
        # No need for manual forward kinematics computation
        if len(env_indices) > 0 and hasattr(self, 'foot_positions'):
            # Store foot positions for planning if needed (already computed in parent class)
            if hasattr(self, 'planned_foot_positions'):
                self.planned_foot_positions[env_indices] = self.foot_positions[env_indices].clone()
            else:
                self.planned_foot_positions = torch.zeros(
                    (self.total_num_envs, self.feet_indices.shape[0], 3), device=self.device
                )
                self.planned_foot_positions[env_indices] = self.foot_positions[env_indices].clone()

    def _reward_gaits(self):
        """Reward for tracking target foot height based on gait pattern."""
        # Get current foot heights
        z_feet = self.foot_positions[:, :, 2]

        # Get parameters for the current gait
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]

        # Calculate target heights for each environment
        # NOTE: main env reward is not used, ignore it
        z_feet_tar = self.get_foot_step(duty_ratio, cadence, amplitude, phases, self.t_rollout)
        z_feet_tar = z_feet_tar.unsqueeze(0).repeat(self.total_num_envs, 1)

        # Compute squared error normalized by tolerance
        error = ((z_feet_tar - z_feet) / 0.05) ** 2
        return -torch.sum(error, dim=1)

    def _reward_feet_contact_forces(self):
        """ElSpider Air specific reward for appropriate foot contact forces."""
        contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        reward = torch.sum((contact_forces > 1.) * (contact_forces < self.cfg.rewards.max_contact_force), dim=1)
        return reward

    def _reward_feet_slip(self):
        """ElSpider Air specific reward to penalize foot slipping."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.total_num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_air_time(self):
        """ElSpider Air specific reward for appropriate foot air time."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * contact_filt, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_base_height(self):
        """ElSpider Air specific reward for maintaining appropriate base height."""
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_collision(self):
        """ElSpider Air specific collision detection and penalty."""
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_async_gait_scheduler(self):
        # Reward for Async Gait Scheduler
        gait_scheduler_scales = class_to_dict(self.cfg.rewards.async_gait_scheduler)

        def get_weight(key, stage):
            if isinstance(gait_scheduler_scales[key], list):
                return gait_scheduler_scales[key][min(stage, len(gait_scheduler_scales[key])-1)]
            else:
                return gait_scheduler_scales[key]

        return self.async_gait_scheduler.reward_dof_align()*get_weight('dof_align', self.reward_scales_stage) + \
            self.async_gait_scheduler.reward_dof_nominal_pos()*get_weight('dof_nominal_pos', self.reward_scales_stage) + \
            self.async_gait_scheduler.reward_foot_z_align()*get_weight('reward_foot_z_align', self.reward_scales_stage)


    def compute_observations(self):
        """Override to provide ElSpider Air specific observations for planning."""
        # Use the parent class observation computation
        super().compute_observations()

        # Add ElSpider Air specific observations if needed
        # For planning, we might want to include predicted foot positions
        if hasattr(self, 'planned_foot_positions') and self.cfg.planning.include_foot_predictions == True:
            foot_obs = self.planned_foot_positions.view(self.total_num_envs, -1)
            self.obs_buf = torch.cat((self.obs_buf, foot_obs), dim=-1)

    def reset_idx(self, env_ids):
        """Override to add ElSpider Air specific reset logic."""
        super().reset_idx(env_ids)

        # Reset ElSpider Air specific states
        if hasattr(self, 'planned_foot_positions'):
            self.planned_foot_positions[env_ids] = 0.
