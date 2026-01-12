# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from typing import Tuple

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import quat_apply
from isaacgym.torch_utils import torch_rand_float
from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymtorch


class Franka(LeggedRobot):
    """Franka Panda robot arm environment for manipulation tasks.
    
    This class extends LeggedRobot but is designed for a fixed-base robot arm.
    The robot performs manipulation tasks with obstacle avoidance and 
    trajectory planning objectives.
    """
    

    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize Franka environment.
        
        Args:
            cfg: Configuration object for the Franka environment
            sim_params: Simulation parameters
            physics_engine: Physics engine to use  
            sim_device: Device for simulation
            headless: Whether to run without rendering
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize end-effector target
        self.ee_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_target_quat[:, 3] = 1.0  # Initialize as identity quaternion
        
        # Initialize current end-effector pose
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_quat[:, 3] = 1.0
        
        # Target reaching tolerance
        self.ee_pos_tolerance = getattr(self.cfg.rewards, 'ee_pos_tolerance', 0.05)
        self.ee_rot_tolerance = getattr(self.cfg.rewards, 'ee_rot_tolerance', 0.1)
        
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()
        
        # Override feet-related buffers since this is an arm robot
        # We'll track end-effector instead
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.ee_quat = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.ee_lin_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.ee_ang_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        # Find end-effector body index
        try:
            self.ee_body_idx = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.cfg.asset.ee_name
            )
        except:
            self.ee_body_idx = -1  # Will use last body if ee_name not found
            print(f"Warning: End-effector body '{getattr(self.cfg.asset, 'ee_name', 'panda_hand')}' not found, using last body")
        
    def post_physics_step(self):
        """Process physics step and update end-effector state."""
        super().post_physics_step()
        
        # Update end-effector state
        self._update_ee_state()
        
    def _update_ee_state(self):
        """Update end-effector position and orientation."""
        if self.ee_body_idx >= 0:
            # Get end-effector state from rigid body state tensor
            self.ee_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 0:3]
            self.ee_quat = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 3:7]
            self.ee_lin_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 7:10]
            self.ee_ang_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 10:13]
        else:
            # Fallback: use last body
            self.ee_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, -1, 0:3]
            self.ee_quat = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, -1, 3:7]
            self.ee_lin_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, -1, 7:10]
            self.ee_ang_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, -1, 10:13]

    def compute_observations(self):
        """Compute observations for robot arm including end-effector pose and target."""
        self.obs_buf = torch.cat((
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.ee_pos,  # Current end-effector position
            self.ee_quat,  # Current end-effector orientation  
            self.ee_target_pos,  # Target end-effector position
            self.ee_target_quat,  # Target end-effector orientation
            self.ee_pos - self.ee_target_pos,  # Position error
        ), dim=-1)

        # Add noise if configured
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        """Resample end-effector target commands for specified environments."""
        # Check if we have external commands to use
        if hasattr(self, 'commands') and self.commands is not None:
            # Use external commands (from trajectory optimization)
            self.ee_target_pos[env_ids] = self.commands[env_ids, :3]
            self.ee_target_quat[env_ids] = self.commands[env_ids, 3:7]
        else:
            # Sample random target positions within workspace
            workspace_min = getattr(self.cfg.commands.ranges, 'ee_pos_min', [-0.5, -0.5, 0.2])
            workspace_max = getattr(self.cfg.commands.ranges, 'ee_pos_max', [0.5, 0.5, 0.8])
            
            for i, dim in enumerate(['x', 'y', 'z']):
                self.ee_target_pos[env_ids, i] = torch_rand_float(
                    workspace_min[i], workspace_max[i], (len(env_ids), 1), device=self.device
                ).squeeze(1)
            
            # Sample random target orientations (simplified - keep orientation roughly upright)
            if getattr(self.cfg.commands, 'randomize_ee_orientation', False):
                # Small random rotations around upright pose
                angle_range = getattr(self.cfg.commands.ranges, 'ee_rot_range', 0.2)
                random_angles = torch_rand_float(-angle_range, angle_range, (len(env_ids), 3), device=self.device)
                
                # Convert to quaternions (simplified - just random rotations)
                self.ee_target_quat[env_ids, :3] = random_angles * 0.5
                self.ee_target_quat[env_ids, 3] = 1.0
                # Normalize quaternions
                self.ee_target_quat[env_ids] = self.ee_target_quat[env_ids] / torch.norm(
                    self.ee_target_quat[env_ids], dim=1, keepdim=True
                )
            else:
                # Keep target orientation as identity (upright)
                self.ee_target_quat[env_ids] = 0.0
                self.ee_target_quat[env_ids, 3] = 1.0
    
    def set_commands(self, commands):
        """Set external commands for end-effector targets.
        
        Args:
            commands: Tensor of shape [num_envs, 7] containing [x, y, z, qx, qy, qz, qw]
        """
        self.commands = commands
        if commands is not None:
            self.ee_target_pos = commands[:, :3]
            self.ee_target_quat = commands[:, 3:7]

    def check_termination(self):
        """Check if environments need to be reset."""
        # Reset based on time limit and other failure conditions
        self.reset_buf = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ) > 1., dim=1)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _reset_root_states(self, env_ids):
        """Reset ROOT states position and velocities of selected environments"""
        # Keep base position fixed for arm robot
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # Zero velocities for fixed base
        self.root_states[env_ids, 7:13] = 0.
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    # Reward functions for robot arm manipulation
    def _reward_ee_position_tracking(self):
        """Reward for end-effector position tracking."""
        pos_error = torch.norm(self.ee_pos - self.ee_target_pos, dim=1)
        return torch.exp(-pos_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_ee_orientation_tracking(self):
        """Reward for end-effector orientation tracking."""
        # Compute quaternion difference
        quat_diff = self.ee_quat - self.ee_target_quat
        rot_error = torch.norm(quat_diff, dim=1)
        return torch.exp(-rot_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_joint_vel(self):
        """Penalize joint velocities for smooth motion."""
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_joint_acc(self):
        """Penalize joint accelerations for smooth motion."""
        return torch.sum(torch.square(self.dof_vel - self.last_dof_vel), dim=1)
    
    def _reward_action_rate(self):
        """Penalize changes in actions for smooth control."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_torques(self):
        """Penalize large torques for energy efficiency."""
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_ee_velocity(self):
        """Penalize large end-effector velocities."""
        return torch.sum(torch.square(self.ee_lin_vel), dim=1) + torch.sum(torch.square(self.ee_ang_vel), dim=1)
    
    def _reward_collision_avoidance(self):
        """Reward for avoiding collisions (using SDF if available)."""
        if hasattr(self, 'sdf_values'):
            # Use SDF values to encourage staying away from obstacles
            min_sdf = torch.min(self.sdf_values, dim=1)[0]
            collision_penalty = torch.clamp(-min_sdf, min=0, max=1.0)  # Penalty when inside obstacles
            return -collision_penalty
        else:
            # Fallback: use contact forces
            contact_penalty = torch.any(torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            ) > 0.1, dim=1).float()
            return -contact_penalty
    
    def _reward_workspace_bounds(self):
        """Penalize end-effector going outside workspace bounds."""
        workspace_min = torch.tensor(
            getattr(self.cfg.commands.ranges, 'ee_pos_min', [-0.6, -0.6, 0.1]), 
            device=self.device
        )
        workspace_max = torch.tensor(
            getattr(self.cfg.commands.ranges, 'ee_pos_max', [0.6, 0.6, 0.9]), 
            device=self.device
        )
        
        out_of_bounds = torch.any(self.ee_pos < workspace_min, dim=1) | torch.any(self.ee_pos > workspace_max, dim=1)
        return -out_of_bounds.float()
    
    def _reward_target_reached(self):
        """Bonus reward for reaching target."""
        pos_error = torch.norm(self.ee_pos - self.ee_target_pos, dim=1)
        target_reached = pos_error < self.ee_pos_tolerance
        return target_reached.float()
