import os
import torch
import numpy as np
import trimesh
import time
from typing import Dict, List, Tuple, Any, Optional, Union

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.envs.batch_rollout.robot_batch_rollout_percept import RobotBatchRolloutPercept
from legged_gym.envs.batch_rollout.robot_traj_grad_sampling_config import RobotTrajGradSamplingCfg
from torch.nn import functional as F

from legged_gym.utils.benchmark import do_cprofile

from traj_sampling.traj_grad_sampling import TrajGradSampling

# Import for RL policy loading
from rsl_rl.modules import ActorCritic
from rsl_rl.modules import ActorCriticRecurrent


class RobotTrajGradSampling(RobotBatchRolloutPercept):
    """Robot environment with trajectory gradient sampling capabilities.

    This class extends RobotBatchRolloutPercept to add:
    1. Storage and management of future trajectories
    2. Optimization of trajectories using sampling-based methods
    3. Evaluation of gradients for trajectory optimization
    4. Optional RL policy warmstart for trajectory initialization
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize environment with trajectory optimization features.

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

        # Initialize action normalization parameters if enabled
        self._init_action_normalization()

        # Initialize trajectory gradient sampling module
        self.traj_grad_sampler = None
        if (hasattr(self.cfg, 'trajectory_opt') and
            getattr(self.cfg.trajectory_opt, "enable_traj_opt", False)) or \
           (hasattr(self.cfg, 'rl_warmstart') and
                getattr(self.cfg.rl_warmstart, "enable", False)):

            self.traj_grad_sampler = TrajGradSampling(
                cfg=self.cfg,
                device=self.device,
                num_envs=self.num_envs,
                num_actions=self.num_actions,
                dt=self.dt,
                main_env_indices=self.main_env_indices
            )

            # Initialize RL policy if enabled
            if hasattr(self.cfg, 'rl_warmstart') and getattr(self.cfg.rl_warmstart, "enable", False):
                self.traj_grad_sampler.init_rl_policy(
                    num_obs=self.num_obs,
                    num_privileged_obs=self.num_privileged_obs if hasattr(self, 'num_privileged_obs') else None
                )

    def _init_trajectories_from_rl(self):
        """Initialize trajectories by rolling out the RL policy."""
        if self.traj_grad_sampler is None:
            return

        def rollout_callback(rl_policy, obs_mean, obs_var):
            """Callback function to perform RL policy rollout."""
            # Sync the rollout environments with the main environments first
            self._sync_main_to_rollout()

            # Create a tensor to hold the action trajectories for all main environments
            batch_size = self.num_envs
            horizon = self.traj_grad_sampler.horizon_samples
            traj_actions = torch.zeros((batch_size, horizon+1, self.traj_grad_sampler.action_size), device=self.device)

            # Get current observations from rollout environments
            self.mean_traj_env_indices = torch.range(0, self.num_envs - 1, device=self.device).long()*self.num_rollout_per_main
            obs_batch = self.obs_buf[self.main_env_indices + 1].clone()
            # Roll out the policy for each step in the horizon
            for i in range(horizon+1):
                # Prepare observations for policy (using privileged or non-privileged based on config)
                if self.cfg.rl_warmstart.obs_type == "privileged" and hasattr(self, "rollout_privileged_obs"):
                    policy_obs = self.rollout_privileged_obs.clone()
                else:
                    policy_obs = obs_batch.clone()

                # Standardize observations if needed
                if obs_mean is not None and obs_var is not None:
                    policy_obs = (policy_obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

                with torch.no_grad():
                    # Get actions from policy using act_inference method
                    actions = rl_policy.act_inference(policy_obs)

                # Store actions in trajectory
                traj_actions[:, i, :] = actions

                # Step rollout environments to get next observations
                obs, privileged_obs, _, _, _ = self.step_rollout(actions)
                obs_batch = obs[self.mean_traj_env_indices].clone()
            # Reset rollout environments back to main state
            self._sync_main_to_rollout()

            return traj_actions

        self.traj_grad_sampler.init_trajectories_from_rl(rollout_callback)

    # Delegate trajectory-related properties to the traj_grad_sampler
    @property
    def node_trajectories(self):
        return self.traj_grad_sampler.node_trajectories if self.traj_grad_sampler else None

    @node_trajectories.setter
    def node_trajectories(self, value):
        if self.traj_grad_sampler:
            self.traj_grad_sampler.node_trajectories = value

    @property
    def action_trajectories(self):
        return self.traj_grad_sampler.action_trajectories if self.traj_grad_sampler else None

    @action_trajectories.setter
    def action_trajectories(self, value):
        if self.traj_grad_sampler:
            self.traj_grad_sampler.action_trajectories = value

    @property
    def predicted_states(self):
        return getattr(self.traj_grad_sampler, 'predicted_states', None) if self.traj_grad_sampler else None

    def node2u(self, nodes: torch.Tensor) -> torch.Tensor:
        """Convert control nodes to dense control sequence using interpolation."""
        if self.traj_grad_sampler:
            return self.traj_grad_sampler.node2u(nodes)
        return nodes

    def node2u_batch(self, nodes_batch: torch.Tensor) -> torch.Tensor:
        """Convert multiple control nodes to dense control sequences at once."""
        if self.traj_grad_sampler:
            return self.traj_grad_sampler.node2u_batch(nodes_batch)
        return nodes_batch

    def u2node(self, us: torch.Tensor) -> torch.Tensor:
        """Convert dense control sequence to control nodes using interpolation."""
        if self.traj_grad_sampler:
            return self.traj_grad_sampler.u2node(us)
        return us

    def u2node_batch(self, us_batch: torch.Tensor) -> torch.Tensor:
        """Convert multiple dense control sequences to control nodes at once."""
        if self.traj_grad_sampler:
            return self.traj_grad_sampler.u2node_batch(us_batch)
        return us_batch

    def shift_nodetraj_batch(self, trajs: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Shift multiple trajectories by n time steps in a batch."""
        if not self.traj_grad_sampler:
            return trajs

        # Prepare policy observations if needed
        policy_obs = None
        if (self.traj_grad_sampler.use_rl_warmstart and
            self.traj_grad_sampler.cfg.rl_warmstart.use_for_append and
            self.traj_grad_sampler.rl_policy is not None and
                self.traj_grad_sampler.rl_traj_initialized):

            if self.traj_grad_sampler.cfg.rl_warmstart.obs_type == "privileged" and hasattr(self, "last_mean_traj_privileged_obs"):
                policy_obs = self.last_mean_traj_privileged_obs.clone()
            elif hasattr(self, "last_mean_traj_obs"):
                policy_obs = self.last_mean_traj_obs.clone()

        return self.traj_grad_sampler.shift_nodetraj_batch(trajs, n_steps, policy_obs)

    def shift_trajectory_batch(self) -> None:
        """Update the node trajectories for all environments based on new actions."""
        if not self.traj_grad_sampler:
            return

        # Prepare policy observations if needed
        policy_obs = None
        if (self.traj_grad_sampler.use_rl_warmstart and
            self.traj_grad_sampler.cfg.rl_warmstart.use_for_append and
            self.traj_grad_sampler.rl_policy is not None and
                self.traj_grad_sampler.rl_traj_initialized):

            if self.traj_grad_sampler.cfg.rl_warmstart.obs_type == "privileged" and hasattr(self, "last_mean_traj_privileged_obs"):
                policy_obs = self.last_mean_traj_privileged_obs.clone()
            elif hasattr(self, "last_mean_traj_obs"):
                policy_obs = self.last_mean_traj_obs.clone()

        self.traj_grad_sampler.shift_trajectory_batch(policy_obs)

    def eval_all_traj_grad(self,
                           mean_trajs: torch.Tensor,
                           noise_scale: Optional[torch.Tensor] = None,
                           n_samples: Optional[int] = None) -> torch.Tensor:
        """Evaluate trajectory gradients for all main environments simultaneously using batch processing."""
        if not self.traj_grad_sampler:
            return mean_trajs

        def rollout_callback(us_batch):
            """Callback function to perform batch rollout."""
            return self.rollout_batch(us_batch)

        return self.traj_grad_sampler.eval_all_traj_grad(
            mean_trajs, rollout_callback, noise_scale, n_samples
        )

    def optimize_all_trajectories(self,
                                  n_diffuse: Optional[int] = None,
                                  initial: bool = False) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Optimize trajectories for all main environments in batch."""
        if not self.traj_grad_sampler:
            return []

        # Initialize trajectories using RL policy if enabled
        if (self.traj_grad_sampler.use_rl_warmstart and
            hasattr(self.traj_grad_sampler, 'node_trajectories') and
                not self.traj_grad_sampler.rl_traj_initialized):
            self._init_trajectories_from_rl()

        def rollout_callback(us_batch):
            """Callback function to perform batch rollout."""
            return self.rollout_batch(us_batch)

        self.traj_grad_sampler.optimize_all_trajectories(rollout_callback, n_diffuse, initial)

        # TODO: Return proper results
        results = []
        return results

    def rollout_batch(self, all_us: torch.Tensor) -> torch.Tensor:
        """Roll out a batch of control sequences using rollout environments."""
        batch_size = all_us.shape[0]
        horizon = all_us.shape[1]
        device = all_us.device

        # Initialize rewards storage
        all_rewards = torch.zeros((batch_size, horizon), device=device)

        # Make sure rollout environments are synchronized with main environment
        self._sync_main_to_rollout()

        # Rollout for each time step in the horizon
        for i in range(horizon):
            # Step all rollout environments with the batch of actions
            obs, privileged_obs, rollout_rewards, dones, infos = self.step_rollout(all_us[:, i, :])
            all_rewards[:, i] = rollout_rewards

        # Store observations for RL policy if needed
        if (self.traj_grad_sampler and
            self.traj_grad_sampler.use_rl_warmstart and
            self.traj_grad_sampler.cfg.rl_warmstart.use_for_append and
                self.traj_grad_sampler.rl_policy is not None):

            if self.traj_grad_sampler.cfg.rl_warmstart.obs_type == "privileged":
                self.last_mean_traj_privileged_obs = privileged_obs[self.mean_traj_env_indices].clone()
            else:
                self.last_mean_traj_obs = obs[self.mean_traj_env_indices].clone()

        # Sync the main environment with the rollout environments
        self._sync_main_to_rollout()
        return all_rewards

    def _init_action_normalization(self):
        """Initialize action normalization parameters if enabled."""
        self.use_action_normalization = getattr(self.cfg.control, 'jointpos_action_normalization', False)

        if self.use_action_normalization:
            # Get joint limits from the asset
            dof_props = self.gym.get_actor_dof_properties(self.envs[0], 0)

            # Extract joint limits
            self.joint_lower_limits = torch.zeros(self.num_actions, device=self.device)
            self.joint_upper_limits = torch.zeros(self.num_actions, device=self.device)

            for i in range(self.num_actions):
                self.joint_lower_limits[i] = dof_props['lower'][i].item() - self.default_dof_pos[0][i].item()
                self.joint_upper_limits[i] = dof_props['upper'][i].item() - self.default_dof_pos[0][i].item()

            # Calculate joint ranges for normalization
            self.joint_ranges = self.joint_upper_limits - self.joint_lower_limits
            self.joint_mid_points = (self.joint_upper_limits + self.joint_lower_limits) / 2.0

            print(f"Action normalization enabled:")
            print(f"Joint lower limits: {self.joint_lower_limits}")
            print(f"Joint upper limits: {self.joint_upper_limits}")
            print(f"Joint ranges: {self.joint_ranges}")

    def _normalize_actions(self, joint_targets: torch.Tensor) -> torch.Tensor:
        """Convert joint position targets to normalized actions [-1, 1].

        Args:
            joint_targets: Joint position targets in radians/meters

        Returns:
            Normalized actions in range [-1, 1]
        """
        if not self.use_action_normalization:
            return joint_targets

        # Convert joint targets to normalized actions
        normalized_actions = 2.0 * (joint_targets - self.joint_lower_limits) / self.joint_ranges - 1.0

        # Clamp to ensure values stay in [-1, 1]
        normalized_actions = torch.clamp(normalized_actions, -1.0, 1.0)

        return normalized_actions

    def _denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions [-1, 1] to joint position targets.

        Args:
            normalized_actions: Normalized actions in range [-1, 1]

        Returns:
            Joint position targets in radians/meters
        """
        if not self.use_action_normalization:
            return normalized_actions

        # Clamp normalized actions to ensure they're in valid range
        normalized_actions = torch.clamp(normalized_actions, -1.0, 1.0)

        # Convert normalized actions to joint targets
        joint_targets = self.joint_lower_limits + (normalized_actions + 1.0) * self.joint_ranges / 2.0

        return joint_targets

    @do_cprofile("./results/step.prof")
    def step(self, actions):
        """Extend step function to handle action normalization and update trajectories."""
        # Denormalize actions if normalization is enabled
        if self.use_action_normalization:
            actions = self._denormalize_actions(actions)

        # Execute the base class step
        obs, priv_obs, rewards, dones, infos = super().step(actions)

        # Update trajectories for each environment
        if self.cfg.trajectory_opt.enable_traj_opt:
            self.shift_trajectory_batch()

        return obs, priv_obs, rewards, dones, infos

    def step_rollout(self, actions: torch.Tensor, action_noise: Optional[torch.Tensor] = None):
        """Extend step_rollout function to handle action normalization."""
        # Denormalize actions if normalization is enabled
        if self.use_action_normalization:
            actions = self._denormalize_actions(actions)

        # Execute the base class step_rollout
        return super().step_rollout(actions, action_noise)

    def _draw_debug_vis(self):
        """Draw debug visualization including predicted trajectories."""
        # Call parent class visualization
        super()._draw_debug_vis()

        # Only draw if viewer and debug_viz are enabled
        if not (self.viewer and self.debug_viz):
            return

        # Draw trajectory visualization if enabled and predictions exist
        if self.predicted_states and 'pos' in self.predicted_states:
            self._draw_trajectory_debug()

    def _draw_trajectory_debug(self):
        """Draw debug visualization for predicted trajectories."""
        # For each main environment, draw its predicted trajectory
        for k in range(self.num_envs):
            i = self.main_env_indices[k]

            # Draw the predicted position trajectory
            if i < self.predicted_states['pos'].shape[0]:
                positions = self.predicted_states['pos'][i].cpu().numpy()

                # Draw the trajectory points
                self.vis.draw_points(i, positions, color=(0, 0.7, 0.3), size=0.03)

                # Connect the points with lines to form a path
                for j in range(len(positions) - 1):
                    self.vis.draw_line(
                        i,
                        [positions[j], positions[j + 1]],
                        color=(0.2, 0.8, 0.2)
                    )
