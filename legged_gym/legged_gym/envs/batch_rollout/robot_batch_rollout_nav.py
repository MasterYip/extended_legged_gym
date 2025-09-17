# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from isaacgym.torch_utils import *

from .robot_batch_rollout_percept import RobotBatchRolloutPercept
from .robot_batch_rollout_nav_config import RobotBatchRolloutNavCfg, RobotBatchRolloutNavCfgPPO


class RobotBatchRolloutNav(RobotBatchRolloutPercept):
    """Navigation environment with automatic velocity command generation.
    
    This environment extends RobotBatchRolloutPercept to add:
    1. Fixed start position for all environments (supports multiple start positions)
    2. Automatic velocity command calculation towards goal (supports multiple goals)
    3. Goal reaching detection
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Initialize navigation-specific variables before calling parent
        self.goal_reached = None
        self.prev_commands = None
        self.start_positions = None
        self.start_orientations = None
        self.goal_positions = None
        self.cfg = cfg
        self.num_main_envs = cfg.env.num_envs
        self.device = sim_device
        # Process start and goal configurations after initialization
        self._process_start_goal_config()

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         sim_device=sim_device,
                         headless=headless)
        

    def _process_start_goal_config(self):
        """Process start and goal configuration to support multiple positions."""     
        # Helper function to expand single position/orientation to match num_main_envs
        def expand_positions(pos_config, num_main_envs):
            # Check if it's a single position [x, y, z] or multiple [[x1, y1, z1], ...]
            if isinstance(pos_config[0], (int, float)):
                # Single position - repeat for all environments
                positions = [pos_config] * num_main_envs
            else:
                # Multiple positions - handle according to requirements
                if num_main_envs <= len(pos_config):
                    # Use first num_main_envs positions
                    positions = pos_config[:num_main_envs]
                else:
                    # Repeat the last position for remaining environments
                    positions = pos_config + [pos_config[-1]] * (num_main_envs - len(pos_config))
            
            return torch.tensor(positions, device=self.device, dtype=torch.float)
        
        # Process start positions
        self.start_positions = expand_positions(self.cfg.navi_opt.start_pos, self.num_main_envs)
        
        # Process start orientations
        self.start_orientations = expand_positions(self.cfg.navi_opt.start_quat, self.num_main_envs)
        
        # Process goal positions
        self.goal_positions = expand_positions(self.cfg.navi_opt.goal_pos, self.num_main_envs)
        
        print(f"Navigation configured with {self.num_main_envs} main environments:")
        print(f"  Start positions shape: {self.start_positions.shape}")
        print(f"  Goal positions shape: {self.goal_positions.shape}")

    def _create_envs(self):
        """Override to set different origins for different environments."""
        # ...existing code from parent...
        super()._create_envs()
        
        # Environment ordering: [main0, rollout0_0, rollout0_1, main1, rollout1_0, rollout1_1, ...]
        # Set environment origins based on their main environment parent
        for i in range(self.total_num_envs):
            # Calculate which main environment this belongs to
            main_env_idx = i // (1 + self.num_rollout_per_main)
            
            # Get start position for this main environment
            start_pos = self.start_positions[main_env_idx]
            self.env_origins[i] = start_pos

    def reset_idx(self, env_ids):
        """Reset environments and initialize navigation state."""
        super().reset_idx(env_ids)
        
        # For each environment being reset, determine its main environment and set appropriate start position
        for env_id in env_ids:
            # Calculate which main environment this belongs to
            main_env_idx = env_id // (1 + self.num_rollout_per_main)
            
            # Get start position and orientation for this main environment
            start_pos = self.start_positions[main_env_idx]
            start_quat = self.start_orientations[main_env_idx]
            
            self.root_states[env_id, 0:3] = start_pos
            self.root_states[env_id, 3:7] = start_quat
            self.root_states[env_id, 7:13] = 0.0  # Zero velocities
        
        # Reset goal reached flag
        if self.goal_reached is None:
            self.goal_reached = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self.goal_reached[env_ids] = False
        
        # Initialize previous commands for smoothing
        if self.prev_commands is None:
            self.prev_commands = torch.zeros(self.total_num_envs, 3, device=self.device)
        self.prev_commands[env_ids] = 0.0

    def _post_physics_step_callback(self):
        """Update navigation state and compute velocity commands."""
        super()._post_physics_step_callback()
        
        # Update velocity commands based on navigation goal
        self._update_navigation_commands()
        
        # Check for goal reaching
        self._check_goal_reached()

    def _post_physics_step_callback_rollout(self):
        """Update navigation state for rollout environments."""
        super()._post_physics_step_callback_rollout()
        
        # Update velocity commands based on navigation goal
        self._update_navigation_commands()
        
        # Check for goal reaching
        self._check_goal_reached()

    def _update_navigation_commands(self):
        """Compute velocity commands to navigate towards the goal."""
        # Get current robot position and orientation
        current_pos = self.root_states[:, 0:3]
        current_quat = self.root_states[:, 3:7]
        
        # Get goal positions for each environment based on their main environment
        goal_pos = torch.zeros(self.total_num_envs, 3, device=self.device)
        
        for i in range(self.total_num_envs):
            # Calculate which main environment this belongs to
            main_env_idx = i // (1 + self.num_rollout_per_main)
            goal_pos[i] = self.goal_positions[main_env_idx]
        
        # Calculate position error
        if self.cfg.navi_opt.use_2d_nav:
            pos_error = goal_pos[:, 0:2] - current_pos[:, 0:2]
            distance_to_goal = torch.norm(pos_error, dim=1)
        else:
            pos_error = goal_pos - current_pos
            distance_to_goal = torch.norm(pos_error, dim=1)
        
        # Calculate desired linear velocity in world frame
        desired_vel_world = torch.zeros(self.total_num_envs, 3, device=self.device)
        
        if self.cfg.navi_opt.use_2d_nav:
            # 2D navigation
            desired_vel_world[:, 0:2] = self.cfg.navi_opt.kp_linear * pos_error
            # Clip to max velocity
            vel_magnitude = torch.norm(desired_vel_world[:, 0:2], dim=1)
            scale_factor = torch.clamp(self.cfg.navi_opt.max_linear_vel / (vel_magnitude + 1e-8), max=1.0)
            desired_vel_world[:, 0:2] *= scale_factor.unsqueeze(1)
        else:
            # 3D navigation
            desired_vel_world = self.cfg.navi_opt.kp_linear * pos_error
            vel_magnitude = torch.norm(desired_vel_world, dim=1)
            scale_factor = torch.clamp(self.cfg.navi_opt.max_linear_vel / (vel_magnitude + 1e-8), max=1.0)
            desired_vel_world *= scale_factor.unsqueeze(1)
        
        # Transform desired velocity to robot frame
        desired_vel_robot = quat_rotate_inverse(current_quat, desired_vel_world)
        
        # Calculate desired angular velocity (towards goal)
        if self.cfg.navi_opt.use_2d_nav:
            # Get current yaw angle
            current_yaw = torch.atan2(2 * (current_quat[:, 3] * current_quat[:, 2] + 
                                          current_quat[:, 0] * current_quat[:, 1]),
                                    1 - 2 * (current_quat[:, 1]**2 + current_quat[:, 2]**2))
            
            # Calculate desired yaw angle towards goal
            desired_yaw = torch.atan2(pos_error[:, 1], pos_error[:, 0])
            
            # Calculate yaw error (handle angle wrapping)
            yaw_error = desired_yaw - current_yaw
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            
            # Calculate desired angular velocity
            desired_angular_vel = self.cfg.navi_opt.kp_angular * yaw_error
            desired_angular_vel = torch.clamp(desired_angular_vel, 
                                            -self.cfg.navi_opt.max_angular_vel, 
                                             self.cfg.navi_opt.max_angular_vel)
        else:
            desired_angular_vel = torch.zeros(self.total_num_envs, device=self.device)
        
        # Create new commands
        new_commands = torch.zeros(self.total_num_envs, 3, device=self.device)
        new_commands[:, 0] = desired_vel_robot[:, 0]  # vx
        new_commands[:, 1] = desired_vel_robot[:, 1]  # vy
        new_commands[:, 2] = desired_angular_vel       # wz
        
        # Apply command smoothing
        if self.prev_commands is not None:
            alpha = self.cfg.navi_opt.cmd_smooth_factor
            smoothed_commands = alpha * self.prev_commands + (1 - alpha) * new_commands
        else:
            smoothed_commands = new_commands
        
        # Update commands
        self.commands[:, 0] = smoothed_commands[:, 0]  # lin_vel_x
        self.commands[:, 1] = smoothed_commands[:, 1]  # lin_vel_y  
        self.commands[:, 2] = smoothed_commands[:, 2]  # ang_vel_yaw
        
        # Store for next iteration
        self.prev_commands = smoothed_commands.clone()
        
        # Stop commanding if goal is reached
        if self.goal_reached is not None:
            self.commands[self.goal_reached] = 0.0

    def _check_goal_reached(self):
        """Check if environments have reached the goal."""
        if self.goal_reached is None:
            self.goal_reached = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        
        # Get current position
        current_pos = self.root_states[:, 0:3]
        
        # Get goal positions for each environment based on their main environment
        goal_pos = torch.zeros(self.total_num_envs, 3, device=self.device)
        
        for i in range(self.total_num_envs):
            # Calculate which main environment this belongs to
            main_env_idx = i // (1 + self.num_rollout_per_main)
            goal_pos[i] = self.goal_positions[main_env_idx]
        
        # Calculate distance to goal
        if self.cfg.navi_opt.use_2d_nav:
            distance = torch.norm(goal_pos[:, 0:2] - current_pos[:, 0:2], dim=1)
        else:
            distance = torch.norm(goal_pos - current_pos, dim=1)
        
        # Update goal reached status
        self.goal_reached = distance < self.cfg.navi_opt.tolerance_rad

    def get_goal_reached_status(self, main_env_only=True):
        """Get goal reached status for all environments.
        
        Returns:
            torch.Tensor: Boolean tensor indicating which environments reached the goal
        """
        if self.goal_reached is None:
            return torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        if main_env_only:
            return self.goal_reached[self.main_env_indices]
        else:
            return self.goal_reached.clone()

    def get_distance_to_goal(self, main_env_only=True):
        """Get distance to goal for all environments.
        
        Returns:
            torch.Tensor: Distance to goal for each environment
        """
        current_pos = self.root_states[:, 0:3]
        
        # Get goal positions for each environment based on their main environment
        goal_pos = torch.zeros(self.total_num_envs, 3, device=self.device)
        
        for i in range(self.total_num_envs):
            # Calculate which main environment this belongs to
            main_env_idx = i // (1 + self.num_rollout_per_main)
            goal_pos[i] = self.goal_positions[main_env_idx]
            
        if self.cfg.navi_opt.use_2d_nav:
            ret = torch.norm(goal_pos[:, 0:2] - current_pos[:, 0:2], dim=1)
        else:
            ret = torch.norm(goal_pos - current_pos, dim=1)
        if main_env_only:
            return ret[self.main_env_indices]
        else:
            return ret.clone()
            
    def _draw_debug_vis(self):
        """Draw debug visualization including navigation elements."""
        super()._draw_debug_vis()
        
        if not (self.viewer and self.debug_viz):
            return
        
        # Draw navigation visualization
        self._draw_navigation_debug()

    def _draw_navigation_debug(self):
        """Draw navigation-specific debug visualization."""
        tolerance_rad = self.cfg.navi_opt.tolerance_rad
        
        for k in range(self.num_main_envs):
            # Get the main environment index (every (1 + rollout_envs) environments)
            i = k * (1 + self.num_rollout_per_main)
            
            # Get goal position for this specific main environment
            goal_pos = self.goal_positions[k].cpu().numpy()
            
            # Draw goal position
            self.vis.draw_point(i, goal_pos, color=(0, 1, 0), size=0.1)
            
            # Draw goal tolerance circle (approximate with line segments)
            num_segments = 16
            for j in range(num_segments):
                angle1 = 2 * np.pi * j / num_segments
                angle2 = 2 * np.pi * (j + 1) / num_segments
                point1 = [goal_pos[0] + tolerance_rad * np.cos(angle1),
                         goal_pos[1] + tolerance_rad * np.sin(angle1),
                         goal_pos[2]]
                point2 = [goal_pos[0] + tolerance_rad * np.cos(angle2),
                         goal_pos[1] + tolerance_rad * np.sin(angle2),
                         goal_pos[2]]
                self.vis.draw_line(i, [point1, point2], color=(0, 1, 0))
            
            # Draw line from robot to goal
            robot_pos = self.root_states[i, 0:3].cpu().numpy()
            self.vis.draw_line(i, [robot_pos, goal_pos], color=(1, 1, 0))
            
            # Draw velocity command vector
            cmd_scale = 0.5
            cmd_vel = self.commands[i, 0:3].cpu().numpy()
            # Transform command to world frame for visualization
            robot_quat = self.root_states[i, 3:7]
            cmd_world = quat_rotate(robot_quat.unsqueeze(0), 
                                   torch.tensor(cmd_vel, device=self.device).unsqueeze(0)).squeeze().cpu().numpy()
            cmd_end = robot_pos + cmd_world * cmd_scale
            self.vis.draw_arrow(i, robot_pos, cmd_end, color=(1, 0, 1), width=0.01)
            
            # Change robot color based on goal reached status
            if self.goal_reached is not None and self.goal_reached[i]:
                # Draw success indicator
                self.vis.draw_point(i, robot_pos + [0, 0, 0.5], color=(0, 1, 0), size=0.15)
