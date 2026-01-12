# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from narwhals import col
import torch
import numpy as np
from typing import Tuple

from legged_gym.envs.batch_rollout.robot_batch_rollout_percept import RobotBatchRolloutPercept
from legged_gym.utils.math_utils import quat_apply
from isaacgym.torch_utils import torch_rand_float
from isaacgym import gymtorch


class FrankaBatchRollout(RobotBatchRolloutPercept):
    """Franka Panda robot arm batch rollout environment for manipulation tasks.
    
    This class extends RobotBatchRolloutPercept but is designed for a fixed-base robot arm.
    The robot performs manipulation tasks with obstacle avoidance and trajectory planning
    objectives using perception features like raycast and SDF.
    """
    

    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize Franka batch rollout environment.
        
        Args:
            cfg: Configuration object for the Franka environment
            sim_params: Simulation parameters
            physics_engine: Physics engine to use  
            sim_device: Device for simulation
            headless: Whether to run without rendering
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize end-effector targets
        self.ee_target_pos = torch.zeros(self.total_num_envs, 3, device=self.device)
        self.ee_target_quat = torch.zeros(self.total_num_envs, 4, device=self.device)
        self.ee_target_quat[:, 3] = 1.0  # Initialize as identity quaternion
        
        # Target reaching tolerance
        self.ee_pos_tolerance = getattr(self.cfg.rewards, 'ee_pos_tolerance', 0.05)
        self.ee_rot_tolerance = getattr(self.cfg.rewards, 'ee_rot_tolerance', 0.1)
        
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()
        
        # Override feet-related buffers since this is an arm robot
        # We'll track end-effector instead
        self.ee_pos = torch.zeros(self.total_num_envs, 3, device=self.device, requires_grad=False)
        self.ee_quat = torch.zeros(self.total_num_envs, 4, device=self.device, requires_grad=False)
        self.ee_lin_vel = torch.zeros(self.total_num_envs, 3, device=self.device, requires_grad=False)
        self.ee_ang_vel = torch.zeros(self.total_num_envs, 3, device=self.device, requires_grad=False)
        
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
        
    def post_physics_step_rollout(self):
        """Process post-physics step for rollout environments only."""
        super().post_physics_step_rollout()
        
        # Update end-effector state for rollout environments
        self._update_ee_state()
        
    def _update_ee_state(self):
        """Update end-effector position and orientation."""
        # Get end-effector state from rigid body state tensor
        self.ee_pos = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 0:3]
        self.ee_quat = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 3:7]
        self.ee_lin_vel = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 7:10]
        self.ee_ang_vel = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.ee_body_idx, 10:13]

    def compute_observations(self):
        """Compute observations for robot arm including end-effector pose and target."""
        # Convert current ee_pos to base-relative coordinates for consistent observations
        ee_pos_relative = self.ee_pos - self.base_pos
        
        self.obs_buf = torch.cat((
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            ee_pos_relative,  # Current end-effector position (base-relative)
            self.ee_quat,  # Current end-effector orientation  
            self.ee_target_pos,  # Target end-effector position (base-relative)
            self.ee_target_quat,  # Target end-effector orientation
            ee_pos_relative - self.ee_target_pos,  # Position error (base-relative)
        ), dim=-1)

        # Add height measurements if configured
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # Add raycast data if enabled and available
        if (hasattr(self, 'ray_caster') and self.ray_caster is not None and 
            hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast and
            hasattr(self, 'raycast_distances') and self.raycast_distances is not None):
            # Add normalized raycast distances to the observations
            self.obs_buf = torch.cat((self.obs_buf, self.raycast_distances), dim=-1)

        # Add SDF values if enabled and configured to be included in observations
        if (hasattr(self, 'mesh_sdf') and self.mesh_sdf is not None and 
            hasattr(self.cfg.sdf, "enable_sdf") and self.cfg.sdf.enable_sdf and 
            self.cfg.sdf.include_in_obs and hasattr(self, 'sdf_values') and self.sdf_values is not None):
            # Add SDF values (flattened) to the observations
            self.obs_buf = torch.cat((self.obs_buf, self.sdf_values), dim=-1)

        # Add noise if configured
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        """Resample end-effector target commands for specified environments."""
        # Check if we have external commands to use
        if hasattr(self, 'commands') and self.commands is not None:
            # Use external commands (from trajectory optimization)
            # Commands are already in base-relative coordinates, use directly
            self.ee_target_pos[env_ids] = self.commands[env_ids, :3]
            self.ee_target_quat[env_ids] = self.commands[env_ids, 3:7]
        else:
            # Sample random target positions within workspace (already base-relative)
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
    
    def set_all_commands(self, commands):
        """Set external commands for end-effector targets.
        
        Args:
            commands: Tensor of shape [total_num_envs, 7] containing [x, y, z, qx, qy, qz, qw]
                     These are already in base-relative coordinates.
        """
        if commands.shape[0] != self.total_num_envs or commands.shape[1] != self.commands.shape[1]:
            raise ValueError(f"Expected commands shape ({self.num_envs}, {self.commands.shape[1]}), got {commands.shape}")

        self.commands[:] = commands
        if commands is not None:
            # Commands are already in base-relative coordinates, use directly
            self.ee_target_pos = self.commands[:, :3]
            self.ee_target_quat = self.commands[:, 3:7]

    def check_termination(self):
        """Check if environments need to be reset."""
        # Reset based on time limit and other failure conditions
        self.reset_buf = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ) > 1., dim=1)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf[self.main_env_indices] |= self.time_out_buf[self.main_env_indices]
        self.target_reach_buf = torch.norm(
            self.ee_pos - self.base_pos - self.ee_target_pos, dim=1
        ) < self.ee_pos_tolerance
        self.reset_buf[self.main_env_indices] |= self.target_reach_buf[self.main_env_indices]

    def _reset_root_states(self, env_ids):
        """Reset ROOT states position and velocities of selected environments"""
        # Keep base position fixed for arm robot
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
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

    def _reset_dofs(self, env_ids):
        """ Reset DOF position and velocities of selected environments """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        """Reset specified environments by resampling commands and resetting root states."""
        if len(env_ids) == self.total_num_envs:
            super().reset_idx(env_ids)
        else:
            print("Franka Experiment don't need to reset.")
        return

    # Reward functions for robot arm manipulation
    def _reward_ee_position_tracking(self):
        """Reward for end-effector position tracking."""
        # Convert current ee_pos to base-relative coordinates for comparison
        ee_pos_relative = self.ee_pos - self.base_pos
        pos_error = torch.norm(ee_pos_relative - self.ee_target_pos, dim=1)
        return torch.exp(-pos_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_ee_orientation_tracking(self):
        """Reward for end-effector orientation tracking."""
        # Compute quaternion difference (orientations are already in correct frame)
        quat_diff = self.ee_quat - self.ee_target_quat
        rot_error = torch.norm(quat_diff, dim=1)
        return torch.exp(-rot_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_dof_vel(self):
        """Penalize joint velocities for smooth motion."""
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
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
        contact_penalty = torch.sum(torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        ), dim=1).float()
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
        
        # Convert current ee_pos to base-relative coordinates for workspace checking
        ee_pos_relative = self.ee_pos - self.base_pos
        out_of_bounds = torch.any(ee_pos_relative < workspace_min, dim=1) | torch.any(ee_pos_relative > workspace_max, dim=1)
        return -out_of_bounds.float()
    
    def _reward_target_reached(self):
        """Bonus reward for reaching target."""
        # Convert current ee_pos to base-relative coordinates for comparison
        ee_pos_relative = self.ee_pos - self.base_pos
        pos_error = torch.norm(ee_pos_relative - self.ee_target_pos, dim=1)
        target_reached =  torch.clamp((self.ee_pos_tolerance - pos_error)/self.ee_pos_tolerance, 0.0) 
        return target_reached.float()

    def _reward_target_reached_dofvel(self):
        """Penalize end-effector velocity & dof vel when target is reached."""
        # Convert current ee_pos to base-relative coordinates for comparison
        ee_pos_relative = self.ee_pos - self.base_pos
        pos_error = torch.norm(ee_pos_relative - self.ee_target_pos, dim=1)
        
        # If within tolerance, penalize velocity
        target_reached = pos_error < self.ee_pos_tolerance*2
        return torch.where(
            target_reached,
            torch.sum(torch.square(self.ee_lin_vel), dim=1) + torch.sum(torch.square(self.ee_ang_vel), dim=1) +
            torch.sum(torch.square(self.dof_vel), dim=1),
            torch.zeros_like(pos_error)
        )

    def _reward_obstacle_avoidance_sdf(self):
        """Reward for collision avoidance using SDF with collision spheres.
        
        Applies penalty when collision spheres penetrate obstacles (SDF < sphere_radius).
        Uses ReLU-style penalty: if sdf_value < sphere_radius, penalty = sdf_value - sphere_radius
        """
        total_penalty = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float)
        
        # Check collision for each query body
        for i in range(self.num_sdf_bodies):
            # Get SDF values for this body across all environments
            sdf_vals = self.sdf_values[:, i]  # Shape: [num_envs]
            
            # Get collision sphere radius for this body
            if (hasattr(self.cfg.sdf, 'collision_sphere_radius') and 
                len(self.cfg.sdf.collision_sphere_radius) > i):
                sphere_radius = self.cfg.sdf.collision_sphere_radius[i]
            else:
                # Default sphere radius if not configured
                sphere_radius = 0.05  # 5cm default
            
            # Calculate penetration depth (negative when sphere penetrates obstacle)
            penetration = sdf_vals - sphere_radius
            
            # Apply ReLU penalty: only penalize when penetrating (penetration < 0)
            collision_penalty = torch.clamp(-penetration, min=0.0)  # ReLU(-penetration)

            # Accumulate penalty from all bodies
            total_penalty += collision_penalty
        
        # Return negative penalty as reward (more penalty = less reward)
        return total_penalty

    def _reward_raycast_obstacle_avoidance(self):
        """Reward for avoiding obstacles based on raycast distances."""
        if hasattr(self, 'raycast_distances') and self.raycast_distances is not None:
            # Encourage maintaining clearance in all directions
            min_clearance = torch.min(self.raycast_distances, dim=1)[0]
            clearance_threshold = getattr(self.cfg.rewards, 'clearance_threshold', 0.3)
            
            # Reward for maintaining good clearance
            clearance_reward = torch.tanh(min_clearance / clearance_threshold)
            return clearance_reward
        else:
            return torch.zeros(self.total_num_envs, device=self.device)

    def _draw_debug_vis(self):
        """ Draws visualizations for debugging (slows down simulation a lot).
            Default behaviour: draws height measurement points and goal spheres for Franka.
        """
        # Call parent class debug visualization
        super()._draw_debug_vis()
        
        # Draw goal points as green spheres for Franka end-effector targets
        if hasattr(self, 'ee_target_pos') and self.ee_target_pos is not None:
            from isaacgym import gymutil, gymapi
            
            # Debug print for first environment
            # if self.num_main_envs > 0:
            #     i = self.main_env_indices[0]
            #     ee_pos_relative = self.ee_pos[i] - self.base_pos[i]
            #     print(f"Debug - Env {i}:")
            #     print(f"  EE pos (world): {self.ee_pos[i]}")
            #     print(f"  EE pos (base-rel): {ee_pos_relative}")
            #     print(f"  EE target (base-rel): {self.ee_target_pos[i]}")
            #     print(f"  Env origin: {self.base_pos[i]}")
            #     print(f"  Position error: {torch.norm(ee_pos_relative - self.ee_target_pos[i])}")
            
            # Create green sphere geometry for goal visualization
            goal_sphere_geom = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(0, 1, 0))  # Green spheres
            
            # Draw goal spheres for main environments only (to reduce visual clutter)
            for j in range(self.num_main_envs):
                i = self.main_env_indices[j]
                
                # Get goal position for this environment (base-relative)
                goal_pos = self.ee_target_pos[i].cpu().numpy()
                
                # Add environment origin offset to get world coordinates for visualization
                world_goal_pos = goal_pos + self.base_pos[i].cpu().numpy()

                
                # Draw the goal sphere
                self.vis.draw_point(0, world_goal_pos, (0,1,0), 0.07)
                self.vis.draw_point(0, self.ee_pos[i], (1,0,0), 0.07)
