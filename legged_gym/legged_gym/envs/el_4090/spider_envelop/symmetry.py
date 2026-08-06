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

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
# from .mixed_terrains.elspider_air_rough_config import ElSpiderAirRoughCfg
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler, \
    SimpleRaibertPlannerConfig, SimpleRaibertPlanner, RaibertPlanner, RaibertPlannerConfig
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw


def _height_measurements_start(obs: torch.Tensor, env=None) -> int:
    """Return the first perception index, keeping structure condition in command obs."""
    if env is not None:
        return _obs_layout(env)["perception_start"]
    return 83 if obs.shape[1] in (83, 270) else 81 if obs.shape[1] in (81, 268) else 74 if obs.shape[1] in (74, 261) else 77 if obs.shape[1] in (77, 264) else 75 if obs.shape[1] in (75, 262) else 72 if obs.shape[1] in (72, 259) else 69 if obs.shape[1] in (69, 256) else 67 if obs.shape[1] in (67, 254) else 66


def _condition_dim(env=None) -> int:
    if env is not None:
        cfg = getattr(env, "cfg", None)
        commands_cfg = getattr(cfg, "commands", None)
        return int(getattr(commands_cfg, "condition_dim", 0))
    return 8


def _obs_layout(env=None) -> Dict[str, int]:
    condition_dim = _condition_dim(env)
    dof_pos_start = 12 + condition_dim
    dof_vel_start = dof_pos_start + 18
    actions_start = dof_vel_start + 18
    physical_prior_start = actions_start + 18
    physical_prior_dim = 0
    haa_range_obs_dim = 0
    gait_phase_obs_dim = 0
    if env is not None:
        cfg = getattr(env, "cfg", None)
        env_cfg = getattr(cfg, "env", None)
        physical_prior_dim = int(getattr(env_cfg, "num_physical_priors", 0))
        haa_range_obs_dim = int(getattr(env_cfg, "num_haa_range_observations", 0))
        gait_phase_obs_dim = int(getattr(env_cfg, "num_gait_phase_observations", 0))
    haa_range_start = physical_prior_start + physical_prior_dim
    gait_phase_start = haa_range_start + haa_range_obs_dim
    perception_start = gait_phase_start + gait_phase_obs_dim
    return {
        "condition_dim": condition_dim,
        "condition_start": 12,
        "dof_pos_start": dof_pos_start,
        "dof_vel_start": dof_vel_start,
        "actions_start": actions_start,
        "physical_prior_start": physical_prior_start,
        "physical_prior_dim": physical_prior_dim,
        "haa_range_start": haa_range_start,
        "haa_range_obs_dim": haa_range_obs_dim,
        "gait_phase_start": gait_phase_start,
        "gait_phase_obs_dim": gait_phase_obs_dim,
        "perception_start": perception_start,
    }


def _mirror_physical_prior_back_forth(obs_mirrored: torch.Tensor, obs: torch.Tensor, env=None) -> None:
    """Swap the appended front/back morphology priors under x-axis reflection."""
    layout = _obs_layout(env)
    if layout["physical_prior_dim"] != 3:
        return
    start = layout["physical_prior_start"]
    obs_mirrored[:, start] = obs[:, start + 2]
    obs_mirrored[:, start + 1] = obs[:, start + 1]
    obs_mirrored[:, start + 2] = obs[:, start]


def _mirror_haa_ranges(obs_mirrored: torch.Tensor, obs: torch.Tensor, env=None, axis="left_right") -> None:
    """Mirror the six HAA centers and six half-ranges in simulator leg order."""
    layout = _obs_layout(env)
    if layout["haa_range_obs_dim"] != 12:
        return
    if axis == "left_right":
        leg_map = torch.tensor([3, 4, 5, 0, 1, 2], dtype=torch.long, device=obs.device)
    else:
        leg_map = torch.tensor([1, 0, 2, 4, 3, 5], dtype=torch.long, device=obs.device)
    start = layout["haa_range_start"]
    center = obs[:, start:start + 6]
    half_range = obs[:, start + 6:start + 12]
    obs_mirrored[:, start:start + 6] = center.index_select(1, leg_map)
    obs_mirrored[:, start + 6:start + 12] = half_range.index_select(1, leg_map)


def _mirror_gait_phase_left_right(obs_mirrored: torch.Tensor, obs: torch.Tensor, env=None) -> None:
    """Left-right reflection swaps the two tripod groups, i.e. phase += pi."""
    layout = _obs_layout(env)
    if layout["gait_phase_obs_dim"] != 2:
        return
    start = layout["gait_phase_start"]
    obs_mirrored[:, start:start + 2] = -obs[:, start:start + 2]


def _mirror_condition_left_right(obs_mirrored: torch.Tensor, obs: torch.Tensor, env=None) -> None:
    layout = _obs_layout(env)
    start = layout["condition_start"]
    condition_dim = layout["condition_dim"]
    if condition_dim in (8, 9, 11) and obs.shape[1] >= start + condition_dim:
        c = obs[:, start:start + condition_dim]
        mirrored = c.clone()
        if condition_dim == 11:
            mirrored[:, 0] = -c[:, 1]  # left_front <- right_front
            mirrored[:, 1] = -c[:, 0]  # right_front <- left_front
            mirrored[:, 2] = -c[:, 3]  # left_mid <- right_mid
            mirrored[:, 3] = -c[:, 2]  # right_mid <- left_mid
            mirrored[:, 4] = -c[:, 5]  # left_back <- right_back
            mirrored[:, 5] = -c[:, 4]  # right_back <- left_back
        obs_mirrored[:, start:start + condition_dim] = mirrored
    elif condition_dim == 6 and obs.shape[1] >= start + condition_dim:
        obs_mirrored[:, start:start + 3] = obs[:, start + 3:start + 6]
        obs_mirrored[:, start + 3:start + 6] = obs[:, start:start + 3]


def _mirror_condition_back_forth(obs_mirrored: torch.Tensor, obs: torch.Tensor, env=None) -> None:
    layout = _obs_layout(env)
    start = layout["condition_start"]
    condition_dim = layout["condition_dim"]
    if condition_dim in (8, 9, 11) and obs.shape[1] >= start + condition_dim:
        c = obs[:, start:start + condition_dim]
        mirrored = c.clone()
        if condition_dim == 8:
            mirrored[:, 0] = c[:, 2]   # front_width <- back_width
            mirrored[:, 2] = c[:, 0]   # back_width <- front_width
            mirrored[:, 3] = -c[:, 4]  # forward_limit <- backward_limit
            mirrored[:, 4] = -c[:, 3]  # backward_limit <- forward_limit
            mirrored[:, 5] = c[:, 7]   # morphology_front_prior <- back
            mirrored[:, 6] = c[:, 6]   # morphology_middle_prior
            mirrored[:, 7] = c[:, 5]   # morphology_back_prior <- front
        else:
            mirrored[:, 0] = c[:, 4]   # left_front <- left_back
            mirrored[:, 1] = c[:, 5]   # right_front <- right_back
            mirrored[:, 4] = c[:, 0]   # left_back <- left_front
            mirrored[:, 5] = c[:, 1]   # right_back <- right_front
            mirrored[:, 6] = -c[:, 7]  # forward_limit <- backward_limit
            mirrored[:, 7] = -c[:, 6]  # backward_limit <- forward_limit
        if condition_dim == 11:
            mirrored[:, 8] = c[:, 10]  # morphology_front_prior <- back
            mirrored[:, 9] = c[:, 9]   # morphology_middle_prior
            mirrored[:, 10] = c[:, 8]  # morphology_back_prior <- front
        obs_mirrored[:, start:start + condition_dim] = mirrored


def _perception_grid_shape(env=None) -> Tuple[int, int]:
    if env is not None:
        cfg = getattr(env, "cfg", None)
        lidar_cfg = getattr(cfg, "lidar", None)
        if getattr(lidar_cfg, "enable", False):
            return lidar_cfg.vertical_line_num, lidar_cfg.horizontal_line_num
    return 17, 11


def _mirror_height_measurements(obs_mirrored: torch.Tensor, obs: torch.Tensor, start: int, axis: str, env=None) -> None:
    x_points, y_points = _perception_grid_shape(env)
    num_heights = x_points * y_points
    if obs.shape[1] < start + num_heights:
        return

    for x in range(x_points):
        for y in range(y_points):
            original_idx = start + x * y_points + y
            mirrored_x = x_points - x - 1 if axis == "x" else x
            mirrored_y = y_points - y - 1 if axis == "y" else y
            mirrored_idx = start + mirrored_x * y_points + mirrored_y
            obs_mirrored[:, original_idx] = obs[:, mirrored_idx]


def _lidar_grid_shape(env=None) -> Tuple[int, int]:
    if env is not None:
        cfg = getattr(env, "cfg", None)
        lidar_cfg = getattr(cfg, "lidar", None)
        if lidar_cfg is not None:
            return lidar_cfg.vertical_line_num, lidar_cfg.horizontal_line_num
    return 11, 17


def _mirror_lidar_observations(obs_mirrored: torch.Tensor, obs: torch.Tensor, start: int, env=None) -> None:
    """Mirror forward-facing grid LiDAR left-right.

    LidarSensor flattens distances from [vertical_line, horizontal_line].
    Horizontal rays are ordered from +azimuth to -azimuth, so left-right
    mirroring keeps each vertical scan line and reverses the horizontal index.
    """
    vertical_lines, horizontal_lines = _lidar_grid_shape(env)
    num_lidar = vertical_lines * horizontal_lines
    if obs.shape[1] < start + num_lidar:
        return

    lidar = obs[:, start:start + num_lidar].reshape(obs.shape[0], vertical_lines, horizontal_lines)
    obs_mirrored[:, start:start + num_lidar] = torch.flip(lidar, dims=[2]).reshape(obs.shape[0], num_lidar)


def _mirror_elspider_obs_left_right(obs: torch.Tensor, env=None, perception: str = "height") -> torch.Tensor:
    obs_mirrored = obs.clone()
    layout = _obs_layout(env)
    dof_pos_start = layout["dof_pos_start"]
    dof_vel_start = layout["dof_vel_start"]
    actions_start = layout["actions_start"]

    # Mirror linear velocity y-component
    obs_mirrored[:, 1] = -obs[:, 1]

    # Mirror angular velocity x and z components
    obs_mirrored[:, 3] = -obs[:, 3]
    obs_mirrored[:, 5] = -obs[:, 5]

    # Mirror projected gravity y-component
    obs_mirrored[:, 7] = -obs[:, 7]

    # Mirror command velocities (y and angular z)
    obs_mirrored[:, 10] = -obs[:, 10]
    obs_mirrored[:, 11] = -obs[:, 11]

    _mirror_condition_left_right(obs_mirrored, obs, env)
    _mirror_haa_ranges(obs_mirrored, obs, env, axis="left_right")
    _mirror_gait_phase_left_right(obs_mirrored, obs, env)

    # Swap left-right DOF positions, velocities, and previous actions.
    obs_mirrored[:, dof_pos_start:dof_pos_start + 9] = obs[:, dof_pos_start + 9:dof_pos_start + 18]
    obs_mirrored[:, dof_pos_start + 9:dof_pos_start + 18] = obs[:, dof_pos_start:dof_pos_start + 9]
    obs_mirrored[:, dof_vel_start:dof_vel_start + 9] = obs[:, dof_vel_start + 9:dof_vel_start + 18]
    obs_mirrored[:, dof_vel_start + 9:dof_vel_start + 18] = obs[:, dof_vel_start:dof_vel_start + 9]
    obs_mirrored[:, actions_start:actions_start + 9] = obs[:, actions_start + 9:actions_start + 18]
    obs_mirrored[:, actions_start + 9:actions_start + 18] = obs[:, actions_start:actions_start + 9]

    start = _height_measurements_start(obs, env)
    if perception == "lidar":
        _mirror_lidar_observations(obs_mirrored, obs, start, env)
    else:
        _mirror_height_measurements(obs_mirrored, obs, start, axis="y", env=env)

    return obs_mirrored


def _mirror_elspider_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions_mirrored = actions.clone()
    actions_mirrored[:, 0:9] = actions[:, 9:18]
    actions_mirrored[:, 9:18] = actions[:, 0:9]
    return actions_mirrored


def _lidar_enabled(env=None) -> bool:
    if env is None:
        return True
    cfg = getattr(env, "cfg", None)
    lidar_cfg = getattr(cfg, "lidar", None)
    return bool(getattr(lidar_cfg, "enable", False))


@torch.no_grad()
def get_elair_lidar_xsym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply left-right symmetry for ElSpider observations with forward grid LiDAR.

    This is the LiDAR equivalent of get_elair_xsym_obs_act. It mirrors robot
    state/action terms exactly the same way, but treats obs[start:] as a
    [vertical_line_num, horizontal_line_num] LiDAR image and flips only the
    horizontal ray axis. Front-back LiDAR symmetry is intentionally omitted
    because the current sensor only covers the forward FOV. If LiDAR is disabled
    on env, this falls back to the original height-map mirror.
    """
    if obs is not None:
        layout = _obs_layout(env)
        dof_pos_start = layout["dof_pos_start"]
        dof_vel_start = layout["dof_vel_start"]
        actions_start = layout["actions_start"]

        perception = "lidar" if _lidar_enabled(env) else "height"
        obs_mirrored = _mirror_elspider_obs_left_right(obs, env=env, perception=perception)
        obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
    else:
        obs_augmented = None

    if actions is not None:
        actions_mirrored = _mirror_elspider_actions_left_right(actions)
        actions_augmented = torch.cat([actions, actions_mirrored], dim=0)
    else:
        actions_augmented = None

    return obs_augmented, actions_augmented


@torch.no_grad()
def get_elair_xysym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply both left-right and back-forth symmetry transformations for the ElSpider robot.
    
    This function augments the dataset by mirroring the robot's left-right sides and front-back directions.
    Returns [batch*3, dim] where first batch is original, second batch is left-right mirrored, 
    third batch is back-forth mirrored.
    
    Foot index order: LB(0-2), LF(3-5), LM(6-8), RB(9-11), RF(12-14), RM(15-17)
    Robot heading: x-axis forward, y-axis left, -y-axis right
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for reference)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors [batch*3, dim]
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]
    
    if obs is not None:
        layout = _obs_layout(env)
        dof_pos_start = layout["dof_pos_start"]
        dof_vel_start = layout["dof_vel_start"]
        actions_start = layout["actions_start"]

        # --- Left-Right Mirrored Observations ---
        obs_lr_mirrored = obs.clone()
        
        # Mirror linear velocity y-component
        obs_lr_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_lr_mirrored[:, 3] = -obs[:, 3]
        obs_lr_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_lr_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_lr_mirrored[:, 10] = -obs[:, 10]
        obs_lr_mirrored[:, 11] = -obs[:, 11]
        
        _mirror_condition_left_right(obs_lr_mirrored, obs, env)
        _mirror_haa_ranges(obs_lr_mirrored, obs, env, axis="left_right")
        _mirror_gait_phase_left_right(obs_lr_mirrored, obs, env)

        # Swap left-right DOF positions: L(0-8) <-> R(9-17)
        obs_lr_mirrored[:, dof_pos_start:dof_pos_start + 9] = obs[:, dof_pos_start + 9:dof_pos_start + 18]
        obs_lr_mirrored[:, dof_pos_start + 9:dof_pos_start + 18] = obs[:, dof_pos_start:dof_pos_start + 9]
        
        # Mirror DOF velocities (30:48)
        obs_lr_mirrored[:, dof_vel_start:dof_vel_start + 9] = obs[:, dof_vel_start + 9:dof_vel_start + 18]
        obs_lr_mirrored[:, dof_vel_start + 9:dof_vel_start + 18] = obs[:, dof_vel_start:dof_vel_start + 9]
        
        # Mirror previous actions (48:66)
        obs_lr_mirrored[:, actions_start:actions_start + 9] = obs[:, actions_start + 9:actions_start + 18]
        obs_lr_mirrored[:, actions_start + 9:actions_start + 18] = obs[:, actions_start:actions_start + 9]
        
        
        _mirror_height_measurements(
            obs_lr_mirrored,
            obs,
            _height_measurements_start(obs, env),
            axis="y",
            env=env,
        )
        
        # --- Back-Forth Mirrored Observations ---
        obs_bf_mirrored = obs.clone()
        
        # Mirror linear velocity x-component
        obs_bf_mirrored[:, 0] = -obs[:, 0]
        
        # Mirror angular velocity y and z components
        obs_bf_mirrored[:, 4] = -obs[:, 4]
        obs_bf_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity x-component
        obs_bf_mirrored[:, 6] = -obs[:, 6]
        
        # Mirror command velocities (x and angular z)
        obs_bf_mirrored[:, 9] = -obs[:, 9]
        obs_bf_mirrored[:, 11] = -obs[:, 11]
        
        _mirror_condition_back_forth(obs_bf_mirrored, obs, env)
        _mirror_physical_prior_back_forth(obs_bf_mirrored, obs, env)
        _mirror_haa_ranges(obs_bf_mirrored, obs, env, axis="back_forth")

        # Swap back-front DOF positions: Back(LB,RB) <-> Front(LF,RF), LM/RM stay as is.
        obs_bf_mirrored[:, dof_pos_start:dof_pos_start + 3] = obs[:, dof_pos_start + 3:dof_pos_start + 6]
        obs_bf_mirrored[:, dof_pos_start + 3:dof_pos_start + 6] = obs[:, dof_pos_start:dof_pos_start + 3]
        obs_bf_mirrored[:, dof_pos_start + 9:dof_pos_start + 12] = obs[:, dof_pos_start + 12:dof_pos_start + 15]
        obs_bf_mirrored[:, dof_pos_start + 12:dof_pos_start + 15] = obs[:, dof_pos_start + 9:dof_pos_start + 12]
        
        # Mirror DOF velocities (30:48) - swap back-front
        obs_bf_mirrored[:, dof_vel_start:dof_vel_start + 3] = obs[:, dof_vel_start + 3:dof_vel_start + 6]
        obs_bf_mirrored[:, dof_vel_start + 3:dof_vel_start + 6] = obs[:, dof_vel_start:dof_vel_start + 3]
        obs_bf_mirrored[:, dof_vel_start + 9:dof_vel_start + 12] = obs[:, dof_vel_start + 12:dof_vel_start + 15]
        obs_bf_mirrored[:, dof_vel_start + 12:dof_vel_start + 15] = obs[:, dof_vel_start + 9:dof_vel_start + 12]
        
        # Mirror previous actions (48:66) - swap back-front
        obs_bf_mirrored[:, actions_start:actions_start + 3] = obs[:, actions_start + 3:actions_start + 6]
        obs_bf_mirrored[:, actions_start + 3:actions_start + 6] = obs[:, actions_start:actions_start + 3]
        obs_bf_mirrored[:, actions_start + 9:actions_start + 12] = obs[:, actions_start + 12:actions_start + 15]
        obs_bf_mirrored[:, actions_start + 12:actions_start + 15] = obs[:, actions_start + 9:actions_start + 12]
        
        _mirror_height_measurements(
            obs_bf_mirrored,
            obs,
            _height_measurements_start(obs, env),
            axis="x",
            env=env,
        )
        
        # Combine original, left-right mirrored, and back-forth mirrored observations
        obs_augmented = torch.cat([obs, obs_lr_mirrored, obs_bf_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # --- Left-Right Mirrored Actions ---
        # Foot index: LB(0-2), LF(3-5), LM(6-8), RB(9-11), RF(12-14), RM(15-17)
        actions_lr_mirrored = actions.clone()
        
        # Swap left and right legs
        actions_lr_mirrored[:, 0:9] = actions[:, 9:18]   # Left legs get right leg actions
        actions_lr_mirrored[:, 9:18] = actions[:, 0:9]   # Right legs get left leg actions
        
        # --- Back-Forth Mirrored Actions ---
        actions_bf_mirrored = actions.clone()
        
        # Swap back and front legs: LB<->LF, RB<->RF, LM stays as is
        actions_bf_mirrored[:, 0:3] = actions[:, 3:6]    # LB gets LF actions
        actions_bf_mirrored[:, 3:6] = actions[:, 0:3]    # LF gets LB actions
        actions_bf_mirrored[:, 9:12] = actions[:, 12:15] # RB gets RF actions
        actions_bf_mirrored[:, 12:15] = actions[:, 9:12] # RF gets RB actions
        
        # Combine original, left-right mirrored, and back-forth mirrored actions
        actions_augmented = torch.cat([actions, actions_lr_mirrored, actions_bf_mirrored], dim=0)
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented


@torch.no_grad()
def get_elair_xsym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply symmetry transformation to observations and actions for the ElSpider robot.
    
    This function augments the dataset by mirroring the robot's left-right sides.
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for reference)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]
    
    # Original and mirrored observations/actions
    # [batch*2, dim] where first batch is original, second batch is mirrored
    
    if obs is not None:
        # Mirror the observations for ElSpider which has 6 legs
        # For policy observation, the structure is:
        # [0:3] - base_lin_vel (mirror y)
        # [3:6] - base_ang_vel (mirror x, z)
        # [6:9] - projected_gravity (mirror y)
        # [9:12] - commands (mirror y for lin_vel, mirror ang_vel_z)
        # [12:30] - dof_pos (swap left-right sides)
        # [30:48] - dof_vel (swap left-right sides)
        # [48:66] - previous actions (swap left-right sides)
        # [12:12+condition_dim] - envelope condition, mirrored with geometry semantics when present
        # perception starts after commands, DOF state, and previous actions.
        
        # Create mirrored observations
        obs_mirrored = obs.clone()
        
        # Mirror linear velocity y-component
        obs_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_mirrored[:, 3] = -obs[:, 3]
        obs_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_mirrored[:, 10] = -obs[:, 10]
        obs_mirrored[:, 11] = -obs[:, 11]
        
        # Swap left-right DOF positions - ElSpider has 6 legs with 3 DOFs each
        # Right side DOFs: 0-8, Left side DOFs: 9-17

        _mirror_condition_left_right(obs_mirrored, obs, env)
        _mirror_haa_ranges(obs_mirrored, obs, env, axis="left_right")
        _mirror_gait_phase_left_right(obs_mirrored, obs, env)

        # Swap right and left DOF positions
        obs_mirrored[:, dof_pos_start:dof_pos_start + 9] = obs[:, dof_pos_start + 9:dof_pos_start + 18]
        obs_mirrored[:, dof_pos_start + 9:dof_pos_start + 18] = obs[:, dof_pos_start:dof_pos_start + 9]
        
        # Mirror DOF velocities (30:48) using the same mapping as positions
        obs_mirrored[:, dof_vel_start:dof_vel_start + 9] = obs[:, dof_vel_start + 9:dof_vel_start + 18]
        obs_mirrored[:, dof_vel_start + 9:dof_vel_start + 18] = obs[:, dof_vel_start:dof_vel_start + 9]
        
        # Mirror previous actions (48:66) using the same mapping
        obs_mirrored[:, actions_start:actions_start + 9] = obs[:, actions_start + 9:actions_start + 18]
        obs_mirrored[:, actions_start + 9:actions_start + 18] = obs[:, actions_start:actions_start + 9]
        
        _mirror_height_measurements(
            obs_mirrored,
            obs,
            _height_measurements_start(obs, env),
            axis="y",
            env=env,
        )
        
        # Combine original and mirrored observations
        obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # Mirror the actions
        # ElSpider has 18 actions (6 legs × 3 joints)
        # Right legs: 0-8, Left legs: 9-17
        actions_mirrored = actions.clone()
        
        actions_mirrored[:, 0:9] = actions[:, 9:18]  # Right legs get left leg actions
        actions_mirrored[:, 9:18] = actions[:, 0:9]  # Left legs get right leg actions
        
        # Combine original and mirrored actions
        actions_augmented = torch.cat([actions, actions_mirrored], dim=0) if actions is not None else None
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented
