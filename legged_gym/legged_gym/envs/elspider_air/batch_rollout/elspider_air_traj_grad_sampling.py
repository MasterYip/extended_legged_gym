import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.batch_rollout.robot_traj_grad_sampling import RobotTrajGradSampling
from legged_gym.envs.elspider_air.batch_rollout.elspider_air_traj_grad_sampling_config import ElSpiderAirTrajGradSamplingCfg, ElSpiderAirTrajGradSamplingCfgPPO
from legged_gym.utils import AsyncGaitSchedulerCfg, AsyncGaitScheduler, GaitScheduler, GaitSchedulerCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw


class ElSpiderAirTrajGradSampling(RobotTrajGradSampling):
    """ElSpider Air environment with trajectory gradient sampling capabilities.

    This class extends RobotTrajGradSampling to add specific functionality
    for the ElSpider Air robot, including custom reward functions and
    terrain interaction optimizations.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize the environment with ElSpider-specific configuration."""

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         sim_device=sim_device,
                         headless=headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

        # Init gait scheduler
        cfg = GaitSchedulerCfg()
        cfg.dt = self.dt
        cfg.period = 1.4
        cfg.swing_height = 0.07
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

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[66:253] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _draw_debug_vis(self):
        # draw base vel
        super()._draw_debug_vis()
        if hasattr(self, 'vis') and self.vis is not None:
            lin_vel = self.root_states[:, 7:10].cpu().numpy()
            cmd_vel_world = quat_apply_yaw(self.base_quat, self.commands[:, :3]).cpu().numpy()
            cmd_vel_world[:, 2] = 0.0
            for j in range(self.num_envs):
                i = self.main_env_indices[j]
                base_pos = self.root_states[i, :3].cpu().numpy()
                self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
                self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))

    # FIXME: this not support both main and rollout envs
    def post_physics_step_rollout(self):
        super().post_physics_step_rollout()
        # Update gait scheduler
        self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions, env_ids=None):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            # with torch.inference_mode():
            self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale +
                                       self.default_dof_pos - self.dof_pos).flatten()
            self.sea_input[:, 0, 1] = self.dof_vel.flatten()
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            torques = torques.view(self.total_num_envs, self.num_actions)
            if env_ids is not None:
                # Select only the torques for the given env_ids
                return torques[env_ids]
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions, env_ids=env_ids)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        super().check_termination()

        # Add new termination condition - terminate if robot is upside down (z-component of projected gravity > 0)
        self.reset_buf |= (self.projected_gravity[:, 2] > 0)

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

    def _reward_gait_scheduler(self):
        # Reward for tracking the gait scheduler
        return self.gait_scheduler.reward_foot_z_track()

