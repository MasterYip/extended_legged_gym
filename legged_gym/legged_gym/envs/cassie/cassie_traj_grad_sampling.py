import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.batch_rollout.robot_traj_grad_sampling import RobotTrajGradSampling
from legged_gym.envs.cassie.cassie_traj_grad_sampling_config import CassieTrajGradSamplingCfg, CassieTrajGradSamplingCfgPPO
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw


class CassieTrajGradSampling(RobotTrajGradSampling):
    """Cassie robot environment with trajectory gradient sampling capabilities.

    This class extends RobotTrajGradSampling to add specific functionality
    for the Cassie robot, including custom reward functions, trajectory
    optimization, and gait control.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize the environment with Cassie-specific configuration.

        Args:
            cfg: Configuration object for the environment
            sim_params: Simulation parameters
            physics_engine: Physics engine to use
            sim_device: Device for simulation
            headless: Whether to run without rendering
        """
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         sim_device=sim_device,
                         headless=headless)

        # Initialize gait scheduler if needed
        self._init_gait_scheduler()

    def _init_gait_scheduler(self):
        """Initialize gait scheduler for coordinated leg movements."""
        # Set up gait scheduler configuration
        cfg = GaitSchedulerCfg()
        cfg.dt = self.dt
        cfg.period = 1.2  # Gait cycle period in seconds (adjust for Cassie's natural dynamics)
        cfg.swing_height = 0.1  # Maximum foot height during swing phase

        # Initialize the gait scheduler
        self.gait_scheduler = GaitScheduler(self.height_samples,
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

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
        Must be adapted for Cassie's observation structure.

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
        # Joint positions and velocities (adjust indices based on Cassie's DOFs)
        noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # Previous actions (no noise)
        noise_vec[37:49] = 0.

        # Add noise to height measurements if used
        if self.cfg.terrain.measure_heights:
            # Adjust the index based on Cassie's observation structure
            noise_vec[49:170] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _draw_debug_vis(self):
        """Draw debug visualizations for Cassie.
        Extends the base class debug visualization with robot-specific elements.
        """
        # Call parent class visualization
        super()._draw_debug_vis()

        # Add Cassie-specific visualizations if viewer is available
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
                    foot_pos_left = self.foot_positions[i, 0].cpu().numpy()
                    foot_pos_right = self.foot_positions[i, 1].cpu().numpy()

                    # Draw spheres at foot positions
                    self.vis.draw_point(i, foot_pos_left, size=0.03, color=(0, 0, 1))
                    self.vis.draw_point(i, foot_pos_right, size=0.03, color=(1, 0, 1))

    def post_physics_step_rollout(self):
        """Process state after physics step for rollout environments.
        Update gait scheduler and other Cassie-specific components.
        """
        # Call parent class post-step processing
        super().post_physics_step_rollout()

        # Update gait scheduler if initialized
        if hasattr(self, 'gait_scheduler'):
            self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands)

    def reset_idx(self, env_ids):
        """Reset specified environments to initial state.

        Args:
            env_ids: Indices of environments to reset
        """
        # Call parent class reset
        super().reset_idx(env_ids)

        # Reset any Cassie-specific states or buffers here if needed

    def _reward_no_fly(self):
        """Reward for maintaining proper foot contact (not flying).
        Ensures at least one foot is in contact with the ground.
        """
        # Check foot contacts (for left and right feet)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        # Reward for having at least one foot in contact
        at_least_one_contact = torch.sum(1.*contacts, dim=1) >= 1
        return 1.*at_least_one_contact

    def _reward_gait_scheduler(self):
        """Reward for tracking the gait scheduler's foot height targets."""
        if hasattr(self, 'gait_scheduler'):
            return self.gait_scheduler.reward_foot_z_track()
        return torch.zeros(self.num_envs, device=self.device)
