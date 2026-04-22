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
from .mixed_terrains.elspider_air_rough_config import ElSpiderAirRoughCfg
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler, \
    SimpleRaibertPlannerConfig, SimpleRaibertPlanner, RaibertPlanner, RaibertPlannerConfig
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw

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
        
        # Swap left-right DOF positions: L(0-8) <-> R(9-17)
        # LB(12:15) LF(15:18) LM(18:21) <-> RB(21:24) RF(24:27) RM(27:30)
        obs_lr_mirrored[:, 12:21] = obs[:, 21:30]  # Left legs get right leg positions
        obs_lr_mirrored[:, 21:30] = obs[:, 12:21]  # Right legs get left leg positions
        
        # Mirror DOF velocities (30:48)
        obs_lr_mirrored[:, 30:39] = obs[:, 39:48]
        obs_lr_mirrored[:, 39:48] = obs[:, 30:39]
        
        # Mirror previous actions (48:66)
        obs_lr_mirrored[:, 48:57] = obs[:, 57:66]
        obs_lr_mirrored[:, 57:66] = obs[:, 48:57]
        
        # Mirror height measurements (66:253) along y-axis
        if obs.shape[1] > 66:
            height_measurements_start = 66
            x_points = 17
            y_points = 11
            
            for x in range(x_points):
                for y in range(y_points):
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_y = y_points - y - 1
                    mirrored_idx = height_measurements_start + x*y_points + mirrored_y
                    obs_lr_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
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
        
        # Swap back-front DOF positions: Back(LB,RB) <-> Front(LF,RF)
        # LB(12:15) <-> LF(15:18), RB(21:24) <-> RF(24:27), LM stays as is
        obs_bf_mirrored[:, 12:15] = obs[:, 15:18]  # LB gets LF positions
        obs_bf_mirrored[:, 15:18] = obs[:, 12:15]  # LF gets LB positions
        obs_bf_mirrored[:, 21:24] = obs[:, 24:27]  # RB gets RF positions
        obs_bf_mirrored[:, 24:27] = obs[:, 21:24]  # RF gets RB positions
        
        # Mirror DOF velocities (30:48) - swap back-front
        obs_bf_mirrored[:, 30:33] = obs[:, 33:36]  # LB gets LF velocities
        obs_bf_mirrored[:, 33:36] = obs[:, 30:33]  # LF gets LB velocities
        obs_bf_mirrored[:, 39:42] = obs[:, 42:45]  # RB gets RF velocities
        obs_bf_mirrored[:, 42:45] = obs[:, 39:42]  # RF gets RB velocities
        
        # Mirror previous actions (48:66) - swap back-front
        obs_bf_mirrored[:, 48:51] = obs[:, 51:54]  # LB gets LF actions
        obs_bf_mirrored[:, 51:54] = obs[:, 48:51]  # LF gets LB actions
        obs_bf_mirrored[:, 57:60] = obs[:, 60:63]  # RB gets RF actions
        obs_bf_mirrored[:, 60:63] = obs[:, 57:60]  # RF gets RB actions
        
        # Mirror height measurements (66:253) along x-axis
        if obs.shape[1] > 66:
            height_measurements_start = 66
            x_points = 17
            y_points = 11
            
            for x in range(x_points):
                for y in range(y_points):
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_x = x_points - x - 1
                    mirrored_idx = height_measurements_start + mirrored_x*y_points + y
                    obs_bf_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
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
        # [66:253] - height measurements (mirror left-right pattern)
        
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

        # Swap right and left DOF positions
        obs_mirrored[:, 12:21] = obs[:, 21:30]  # Right legs get left leg positions
        obs_mirrored[:, 21:30] = obs[:, 12:21]  # Left legs get right leg positions
        
        # Mirror DOF velocities (30:48) using the same mapping as positions
        obs_mirrored[:, 30:39] = obs[:, 39:48]  # Right legs get left leg velocities
        obs_mirrored[:, 39:48] = obs[:, 30:39]  # Left legs get right leg velocities
        
        # Mirror previous actions (48:66) using the same mapping
        obs_mirrored[:, 48:57] = obs[:, 57:66]  # Right legs get left leg actions
        obs_mirrored[:, 57:66] = obs[:, 48:57]  # Left legs get right leg actions
        
        # Mirror height measurements (66:253) if present
        if obs.shape[1] > 66:
            # The height measurements are in a grid pattern
            # Original grid pattern: measured_points_x × measured_points_y
            # For ElSpider, this is typically 17×11 = 187 points
            
            # We need to mirror the points along the y-axis
            # If we have 17 points in x and 11 in y, the indices form a 17×11 grid
            
            height_measurements_start = 66
            x_points = 17  # Number of points along x-axis (from config)
            y_points = 11  # Number of points along y-axis (from config)
            
            for x in range(x_points):
                for y in range(y_points):
                    # Calculate original and mirrored indices
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_y = y_points - y - 1  # Flip y coordinate
                    mirrored_idx = height_measurements_start + x*y_points + mirrored_y
                    
                    # Swap the height measurements
                    obs_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
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

