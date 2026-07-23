import os
import torch
import numpy as np
import trimesh
import time
from typing import Dict, List, Tuple, Any, Optional, Union

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.envs.batch_rollout.robot_traj_grad_sampling import RobotTrajGradSampling
from torch.nn import functional as F

from legged_gym.utils.benchmark import do_cprofile



class RobotPlanGradSampling(RobotTrajGradSampling):
    """Robot environment with planning-based trajectory gradient sampling.

    This class extends RobotTrajGradSampling to optimize state velocity trajectories
    instead of action trajectories. It uses trajectory integration for rollouts rather
    than physics simulation.

    Key differences from RobotTrajGradSampling:
    1. Optimizes state velocities (24-dim for elspider: 3 lin + 3 ang + 18 joint vel)
    2. Uses trajectory integration instead of physics simulation for rollouts
    3. Only uses physics simulation for viewer updates when needed
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize planning environment with state velocity trajectory optimization.

        Args:
            cfg: Configuration object for the environment
            sim_params: Simulation parameters
            physics_engine: Physics engine to use
            sim_device: Device for simulation
            headless: Whether to run without rendering
        """
        # Initialize base class
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         sim_device=sim_device,
                         headless=headless)

        # Planning-specific initialization
        self.state_vel_dim = self.cfg.planning.state_vel_dim
        self.integration_method = self.cfg.planning.integration_method
        self.use_sim_step_for_viewer = self.cfg.planning.use_sim_step_for_viewer
        self.render_rollouts = self.cfg.viewer.render_rollouts
        # State velocity limits
        self.max_base_lin_vel = self.cfg.planning.max_base_lin_vel
        self.max_base_ang_vel = self.cfg.planning.max_base_ang_vel
        self.max_joint_vel = self.cfg.planning.max_joint_vel

        # Integration parameters
        self.max_integration_step = self.cfg.planning.max_integration_step
        self.enforce_joint_limits = self.cfg.planning.enforce_joint_limits

        # Override trajectory sampler to use state velocities
        if self.traj_grad_sampler:
            # Update action size to state velocity dimension
            self.traj_grad_sampler.action_size = self.state_vel_dim
            self.traj_grad_sampler.num_actions = self.state_vel_dim

            # Reinitialize trajectory storage with new dimensions
            self.traj_grad_sampler.node_trajectories = torch.zeros(
                (self.num_envs, self.traj_grad_sampler.horizon_nodes + 1, self.state_vel_dim),
                device=self.device
            )
            self.traj_grad_sampler.action_trajectories = torch.zeros(
                (self.num_envs, self.traj_grad_sampler.horizon_samples + 1, self.state_vel_dim),
                device=self.device
            )

        # Initialize state storage for integration
        self._init_planning_buffers()

    def _init_planning_buffers(self):
        """Initialize buffers for planning and state integration."""
        # Current state for integration (position, orientation, joint positions)
        self.integration_base_pos = torch.zeros((self.total_num_envs, 3), device=self.device)
        self.integration_base_quat = torch.zeros((self.total_num_envs, 4), device=self.device)
        self.integration_dof_pos = torch.zeros((self.total_num_envs, self.num_dof), device=self.device)

        # State velocities for integration
        self.integration_base_lin_vel = torch.zeros((self.total_num_envs, 3), device=self.device)
        self.integration_base_ang_vel = torch.zeros((self.total_num_envs, 3), device=self.device)
        self.integration_dof_vel = torch.zeros((self.total_num_envs, self.num_dof), device=self.device)

        # Joint limits for constraint enforcement
        if self.enforce_joint_limits:
            self.dof_pos_limits = torch.zeros((self.num_dof, 2), device=self.device)
            for i in range(self.num_dof):
                self.dof_pos_limits[i, 0] = self.cfg.asset.dof_pos_limit_lower[i] if hasattr(
                    self.cfg.asset, 'dof_pos_limit_lower') else -np.pi
                self.dof_pos_limits[i, 1] = self.cfg.asset.dof_pos_limit_upper[i] if hasattr(
                    self.cfg.asset, 'dof_pos_limit_upper') else np.pi

    def _integrate_state_velocities(self, state_vels: torch.Tensor, dt: float, env_indices: Optional[torch.Tensor] = None):
        """Integrate state velocities to update robot states.

        Args:
            state_vels: State velocities [batch_size, 24] where 24 = 3 lin_vel + 3 ang_vel + 18 joint_vel
            dt: Integration time step
            env_indices: Environment indices to update (if None, updates all)
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        batch_size = len(env_indices)

        # Extract velocity components
        base_lin_vel = state_vels[:, :3]  # [batch, 3]
        base_ang_vel = state_vels[:, 3:6]  # [batch, 3]
        joint_vel = state_vels[:, 6:]     # [batch, 18]

        # Clip velocities to limits
        base_lin_vel = torch.clamp(base_lin_vel, -self.max_base_lin_vel, self.max_base_lin_vel)
        base_ang_vel = torch.clamp(base_ang_vel, -self.max_base_ang_vel, self.max_base_ang_vel)
        joint_vel = torch.clamp(joint_vel, -self.max_joint_vel, self.max_joint_vel)

        # Use smaller integration step if needed for stability
        integration_dt = min(dt, self.max_integration_step)
        n_steps = int(np.ceil(dt / integration_dt))
        actual_dt = dt / n_steps

        for step in range(n_steps):
            if self.integration_method == "euler":
                # Euler integration
                # Update base position (in world frame)
                self.integration_base_pos[env_indices] += base_lin_vel * actual_dt

                # Update base orientation using angular velocity
                # Convert angular velocity to quaternion update
                angle = torch.norm(base_ang_vel, dim=1, keepdim=True) * actual_dt
                axis = base_ang_vel / (torch.norm(base_ang_vel, dim=1, keepdim=True) + 1e-8)

                # Create rotation quaternion from angle-axis
                rot_quat = quat_from_angle_axis(angle.squeeze(-1), axis)

                # Apply rotation to current quaternion
                self.integration_base_quat[env_indices] = quat_mul(
                    self.integration_base_quat[env_indices],
                    rot_quat
                )

                # Normalize quaternion
                self.integration_base_quat[env_indices] = self.integration_base_quat[env_indices] / torch.norm(
                    self.integration_base_quat[env_indices], dim=1, keepdim=True
                )

                # Update joint positions
                self.integration_dof_pos[env_indices] += joint_vel* actual_dt

            elif self.integration_method == "rk4":
                # RK4 integration (more accurate but slower)
                # For simplicity, implementing only for base position here
                # Joint positions and orientation can use similar approach
                k1 = base_lin_vel
                k2 = base_lin_vel  # Assuming constant velocity
                k3 = base_lin_vel
                k4 = base_lin_vel

                self.integration_base_pos[env_indices] += (k1 + 2*k2 + 2*k3 + k4) * actual_dt / 6

                # For joints, use Euler for now (can be extended to RK4)
                self.integration_dof_pos[env_indices] += joint_vel * actual_dt

                # For orientation, use Euler (RK4 for quaternions is complex)
                angle = torch.norm(base_ang_vel, dim=1, keepdim=True) * actual_dt
                axis = base_ang_vel / (torch.norm(base_ang_vel, dim=1, keepdim=True) + 1e-8)
                rot_quat = quat_from_angle_axis(angle.squeeze(-1), axis)
                self.integration_base_quat[env_indices] = quat_mul(
                    self.integration_base_quat[env_indices], rot_quat
                )
                self.integration_base_quat[env_indices] = self.integration_base_quat[env_indices] / torch.norm(
                    self.integration_base_quat[env_indices], dim=1, keepdim=True
                )

        # Enforce joint limits if enabled
        if self.enforce_joint_limits:
            self.integration_dof_pos[env_indices] = torch.clamp(
                self.integration_dof_pos[env_indices],
                self.dof_pos_limits[:, 0].unsqueeze(0).repeat(batch_size, 1),
                self.dof_pos_limits[:, 1].unsqueeze(0).repeat(batch_size, 1)
            )

        # Update velocity storage
        self.integration_base_lin_vel[env_indices] = base_lin_vel
        self.integration_base_ang_vel[env_indices] = base_ang_vel
        self.integration_dof_vel[env_indices] = joint_vel

    def _sync_integration_to_sim(self, env_indices: Optional[torch.Tensor] = None):
        """Synchronize integrated states to simulation states.

        Args:
            env_indices: Environment indices to sync (if None, syncs all)
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        # Update simulation states from integration
        self.root_states[env_indices, :3] = self.integration_base_pos[env_indices]
        self.root_states[env_indices, 3:7] = self.integration_base_quat[env_indices]
        self.root_states[env_indices, 7:10] = self.integration_base_lin_vel[env_indices]
        self.root_states[env_indices, 10:13] = self.integration_base_ang_vel[env_indices]

        self.dof_pos[env_indices] = self.integration_dof_pos[env_indices]
        self.dof_vel[env_indices] = self.integration_dof_vel[env_indices]

        # Update other derived states
        self.base_pos[env_indices] = self.integration_base_pos[env_indices]
        self.base_quat[env_indices] = self.integration_base_quat[env_indices]
        self.base_lin_vel[env_indices] = quat_rotate_inverse(
            self.base_quat[env_indices],
            self.integration_base_lin_vel[env_indices]
        )
        self.base_ang_vel[env_indices] = quat_rotate_inverse(
            self.base_quat[env_indices],
            self.integration_base_ang_vel[env_indices]
        )

    def _sync_sim_to_integration(self, env_indices: Optional[torch.Tensor] = None):
        """Synchronize simulation states to integration states.

        Args:
            env_indices: Environment indices to sync (if None, syncs all)
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        # Update integration states from simulation
        self.integration_base_pos[env_indices] = self.root_states[env_indices, :3]
        self.integration_base_quat[env_indices] = self.root_states[env_indices, 3:7]
        self.integration_base_lin_vel[env_indices] = self.root_states[env_indices, 7:10]
        self.integration_base_ang_vel[env_indices] = self.root_states[env_indices, 10:13]

        self.integration_dof_pos[env_indices] = self.dof_pos[env_indices]
        self.integration_dof_vel[env_indices] = self.dof_vel[env_indices]

    def step(self, actions):
        """Override step to use planning integration instead of physics simulation.

        Args:
            actions: State velocity commands [num_main_envs, state_vel_dim]
        """
        # Convert actions to state velocities for main environments
        # state_vels = torch.zeros((self.num_main_envs, self.state_vel_dim), device=self.device)
        # state_vels[self.main_env_indices] = actions
        # TODO: Align procedures with robot_batch_rollout.py

        # Integrate state velocities to update robot states
        self._integrate_state_velocities(actions, self.dt, self.main_env_indices)

        # Sync integration results to simulation states
        self._sync_integration_to_sim(self.main_env_indices)
        self._sync_main_to_rollout()

        # Update other environment states like in the parent class
        self.episode_length_buf[self.main_env_indices] += 1
        self.common_step_counter += 1

        # Compute observations and rewards using current states
        self._update_derived_states(self.main_env_indices)
        self.compute_reward()
        self.compute_observations()

        # Check terminations and resets
        self.check_termination()
        env_ids = self.reset_buf[self.main_env_indices].nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            # Map back to main env indices
            main_env_ids = self.main_env_indices[env_ids]
            self.reset_idx(main_env_ids)

        # Render and update trajectories
        # Optionally step simulation for viewer updates (without physics)
        if self.use_sim_step_for_viewer and not self.headless:
            # Set states in simulator for visualization
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

            # Step simulation once for viewer update (minimal physics)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_physics_step()
            self.render()
            

        self.shift_trajectory_batch()

        # Return observations for main environments
        clip_obs = self.cfg.normalization.clip_observations
        main_obs = torch.clip(self.obs_buf[self.main_env_indices], -clip_obs, clip_obs)

        main_privileged_obs = None
        if self.privileged_obs_buf is not None:
            clip_obs = self.cfg.normalization.clip_observations
            main_privileged_obs = torch.clip(self.privileged_obs_buf[self.main_env_indices], -clip_obs, clip_obs)

        main_rew = self.rew_buf[self.main_env_indices]
        main_reset = self.reset_buf[self.main_env_indices]
        main_extras = {key: val[self.main_env_indices] for key, val in self.extras.items()
                       if isinstance(val, torch.Tensor)}

        self.record_step()
        # Update time counters for main environments
        self.t_main += self.dt
        self.t_rollout = self.t_main


        return main_obs, main_privileged_obs, main_rew, main_reset, main_extras

    def step_rollout(self, rollout_state_vels, noise_scales=None):
        """Override step_rollout to use state velocity integration.

        Args:
            rollout_state_vels: State velocities for rollout environments
            noise_scales: Optional noise scales (for compatibility)
        """
        # Determine the mode based on input shape
        is_legacy_mode = (rollout_state_vels.shape[0] == self.num_main_envs)

        if is_legacy_mode:
            # Legacy mode: expand mean state vels to all rollout envs with noise
            state_vels = torch.zeros((len(self.rollout_env_indices), self.state_vel_dim), device=self.device)

            # For each main environment, assign state velocities to its rollout environments
            rollout_idx = 0
            for i in range(self.num_main_envs):
                for j in range(self.num_rollout_per_main):
                    state_vels[rollout_idx] = rollout_state_vels[i]

                    # Add noise if specified
                    if noise_scales is not None:
                        noise = torch.randn_like(state_vels[rollout_idx]) * noise_scales
                        state_vels[rollout_idx] += noise

                    rollout_idx += 1
        else:
            # New mode: direct batch of state velocities
            state_vels = rollout_state_vels

        # Integrate state velocities for rollout environments
        self._integrate_state_velocities(state_vels, self.dt, self.rollout_env_indices)

        # Sync integration results to simulation states
        self._sync_integration_to_sim(self.rollout_env_indices)

        # Step simulation for viewer updates if needed
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step_rollout()
        # Optionally step simulation for viewer updates (without physics)
        if self.render_rollouts and not self.headless:
            self.render()

        # Update derived states and compute rewards/observations
        self._update_derived_states(self.rollout_env_indices)
        self.compute_reward()
        self.compute_observations()

        # Restore main environment states (freeze them during rollout)
        self._restore_main_env_states()

        # Return observations for rollout environments
        clip_obs = self.cfg.normalization.clip_observations
        rollout_obs = torch.clip(self.obs_buf[self.rollout_env_indices], -clip_obs, clip_obs)

        rollout_privileged_obs = None
        if self.privileged_obs_buf is not None:
            rollout_privileged_obs = torch.clip(self.privileged_obs_buf[self.rollout_env_indices], -clip_obs, clip_obs)

        rollout_rew = self.rew_buf[self.rollout_env_indices]
        rollout_reset = self.reset_buf[self.rollout_env_indices]
        rollout_extras = {key: val[self.rollout_env_indices] for key, val in self.extras.items()
                          if isinstance(val, torch.Tensor)}

        # Update time counters for rollout environments
        self.t_rollout += self.dt

        return rollout_obs, rollout_privileged_obs, rollout_rew, rollout_reset, rollout_extras

    def rollout_batch(self, all_state_vels: torch.Tensor) -> torch.Tensor:
        """Roll out a batch of state velocity sequences using integration.

        Args:
            all_state_vels: Batch of state velocity sequences [batch_size, horizon, state_vel_dim]
        """
        batch_size = all_state_vels.shape[0]
        horizon = all_state_vels.shape[1]
        device = all_state_vels.device

        # Initialize rewards storage
        all_rewards = torch.zeros((batch_size, horizon), device=device)

        # Sync main environments to rollout environments
        self._sync_main_to_rollout()

        # Sync simulation states to integration
        self._sync_sim_to_integration()

        # Rollout for each time step in the horizon
        for i in range(horizon):
            # Integrate state velocities for this time step
            self._integrate_state_velocities(all_state_vels[:, i, :], self.dt, self.rollout_env_indices)

            # Sync to simulation states and update derived quantities
            self._sync_integration_to_sim(self.rollout_env_indices)

            # Optionally step simulation for viewer updates (without physics)
            if self.render_rollouts and not self.headless:
                # Set states in simulator for visualization
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

                # Step simulation once for viewer update (minimal physics)
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.render()

            self._update_derived_states(self.rollout_env_indices)

            # Compute rewards for this step
            self.compute_reward()
            all_rewards[:, i] = self.rew_buf[self.rollout_env_indices]

        # Restore main environment states
        self._sync_main_to_rollout()

        return all_rewards

    def _update_derived_states(self, env_indices: torch.Tensor):
        """Update derived states after integration.

        Args:
            env_indices: Environment indices to update
        """
        # Update base velocities in body frame
        self.base_lin_vel[env_indices] = quat_rotate_inverse(
            self.base_quat[env_indices],
            self.root_states[env_indices, 7:10]
        )
        self.base_ang_vel[env_indices] = quat_rotate_inverse(
            self.base_quat[env_indices],
            self.root_states[env_indices, 10:13]
        )

        # Update projected gravity
        self.projected_gravity[env_indices] = quat_rotate_inverse(
            self.base_quat[env_indices],
            self.gravity_vec[env_indices]
        )

        # Update foot positions and velocities
        # Note: This is simplified and may need more sophisticated kinematic calculation
        # For now, we'll rely on the existing rigid body state updates
        if hasattr(self, 'rigid_body_state'):
            self.foot_positions = self.rigid_body_state.view(
                self.total_num_envs, self.num_bodies, 13
            )[:, self.feet_indices, 0:3]
            self.foot_velocities = self.rigid_body_state.view(
                self.total_num_envs, self.num_bodies, 13
            )[:, self.feet_indices, 7:10]

    def _sync_main_to_rollout(self):
        """Override to also sync integration states."""
        # Call parent method to sync simulation states
        super()._sync_main_to_rollout()

        # Also sync integration states
        if len(self.rollout_env_indices) == 0:
            return

        # Sync integration states from main to rollout environments
        for i in range(self.num_main_envs):
            main_idx = self.main_env_indices[i]
            rollout_indices = self.main_to_rollout_indices[i]

            if len(rollout_indices) > 0:
                self.integration_base_pos[rollout_indices] = self.integration_base_pos[main_idx].clone()
                self.integration_base_quat[rollout_indices] = self.integration_base_quat[main_idx].clone()
                self.integration_base_lin_vel[rollout_indices] = self.integration_base_lin_vel[main_idx].clone()
                self.integration_base_ang_vel[rollout_indices] = self.integration_base_ang_vel[main_idx].clone()
                self.integration_dof_pos[rollout_indices] = self.integration_dof_pos[main_idx].clone()
                self.integration_dof_vel[rollout_indices] = self.integration_dof_vel[main_idx].clone()

    def _cache_main_env_states(self):
        """Override to also cache integration states."""
        # Call parent method to cache simulation states
        super()._cache_main_env_states()

        # Cache integration states
        main_indices = self.main_env_indices

        if not hasattr(self, 'main_env_cache'):
            self.main_env_cache = {}

        # Cache integration states
        self.main_env_cache['integration_base_pos'] = self.integration_base_pos[main_indices].clone()
        self.main_env_cache['integration_base_quat'] = self.integration_base_quat[main_indices].clone()
        self.main_env_cache['integration_base_lin_vel'] = self.integration_base_lin_vel[main_indices].clone()
        self.main_env_cache['integration_base_ang_vel'] = self.integration_base_ang_vel[main_indices].clone()
        self.main_env_cache['integration_dof_pos'] = self.integration_dof_pos[main_indices].clone()
        self.main_env_cache['integration_dof_vel'] = self.integration_dof_vel[main_indices].clone()

    def _restore_main_env_states(self):
        """Override to also restore integration states."""
        # Call parent method to restore simulation states
        super()._restore_main_env_states()

        # Restore integration states
        if not hasattr(self, 'main_env_cache'):
            return

        main_indices = self.main_env_indices

        # Restore integration states
        if 'integration_base_pos' in self.main_env_cache:
            self.integration_base_pos[main_indices] = self.main_env_cache['integration_base_pos'].clone()
            self.integration_base_quat[main_indices] = self.main_env_cache['integration_base_quat'].clone()
            self.integration_base_lin_vel[main_indices] = self.main_env_cache['integration_base_lin_vel'].clone()
            self.integration_base_ang_vel[main_indices] = self.main_env_cache['integration_base_ang_vel'].clone()
            self.integration_dof_pos[main_indices] = self.main_env_cache['integration_dof_pos'].clone()
            self.integration_dof_vel[main_indices] = self.main_env_cache['integration_dof_vel'].clone()

    def reset_idx(self, env_ids):
        """Override to also reset integration states."""
        # Call parent method
        super().reset_idx(env_ids)

        # Reset integration states to match simulation states
        self._sync_sim_to_integration(env_ids)
