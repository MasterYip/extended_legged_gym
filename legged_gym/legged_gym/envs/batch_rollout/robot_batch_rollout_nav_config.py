# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from .robot_batch_rollout_percept_config import RobotBatchRolloutPerceptCfg, RobotBatchRolloutPerceptCfgPPO


class RobotBatchRolloutNavCfg(RobotBatchRolloutPerceptCfg):
    class env(RobotBatchRolloutPerceptCfg.env):
        # Override episode length for navigation tasks
        episode_length_s = 30  # Longer episodes for navigation

    class navi_opt:
        # Start positions for environments
        # Can be a single position [x, y, z] or list of positions [[x1, y1, z1], [x2, y2, z2], ...]
        start_pos = [0.0, 0.0, 0.5]
        
        # Start orientations as quaternions
        # Can be a single quaternion [x, y, z, w] or list of quaternions [[x1, y1, z1, w1], [x2, y2, z2, w2], ...]
        start_quat = [0.0, 0.0, 0.0, 1.0]
        
        # Goal positions for environments
        # Can be a single position [x, y, z] or list of positions [[x1, y1, z1], [x2, y2, z2], ...]
        goal_pos = [5.0, 5.0, 0.5]
        
        # Goal tolerance radius for success detection
        tolerance_rad = 0.5  # meters
        
        # Velocity command parameters
        max_linear_vel = 1.0   # m/s
        max_angular_vel = 1.0  # rad/s
        
        # Navigation control gains
        kp_linear = 1.0    # Proportional gain for linear velocity
        kp_angular = 2.0   # Proportional gain for angular velocity
        
        # Command smoothing factor (0.0 = no smoothing, 1.0 = no update)
        cmd_smooth_factor = 0.1
        
        # Whether to use 2D navigation (ignore z-axis)
        use_2d_nav = True

    class commands(RobotBatchRolloutPerceptCfg.commands):
        # Override command ranges since we'll compute them automatically
        class ranges:
            lin_vel_x = [0.0, 0.0]  # Will be overridden by navigation controller
            lin_vel_y = [0.0, 0.0]  # Will be overridden by navigation controller  
            ang_vel_yaw = [0.0, 0.0]  # Will be overridden by navigation controller
            heading = [0.0, 0.0]


class RobotBatchRolloutNavCfgPPO(RobotBatchRolloutPerceptCfgPPO):
    class runner(RobotBatchRolloutPerceptCfgPPO.runner):
        # Adjust for navigation task
        num_steps_per_env = 32
        max_iterations = 2000
