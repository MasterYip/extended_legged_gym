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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.elspider_air.mixed_terrains.elspider_air_rough_train_config import ElSpiderAirRoughTrainCfg, ElSpiderAirRoughTrainCfgPPO


class ElSpiderAirRoughRaycastCfg(ElSpiderAirRoughTrainCfg):
    class env(ElSpiderAirRoughTrainCfg.env):
        # Update observation space for raycast data
        num_observations = 66 + 512  # May need to adjust based on raycast points

    class terrain(ElSpiderAirRoughTrainCfg.terrain):
        use_terrain_obj = False  # use TerrainObj class to create terrain
        # path to the terrain file
        terrain_file = None

        mesh_type = 'confined_trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 10  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level =0  # starting curriculum state
        terrain_length = 6.
        terrain_width = 6.
        num_rows = 4  # number of terrain rows (levels)
        num_cols = 6  # number of terrain cols (types)
        difficulty_scale = 1.0  # Scale for difficulty in curriculum
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.2]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 0.2, 0.3, 0.3]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class raycaster:
        enable_raycast = True
        terrain_file = None
        # Patterns
        ray_pattern = "spherical2"  # Use spherical pattern
        # Spherical pattern settings
        spherical_num_azimuth = 12  # Number of rays in horizontal (azimuth) direction
        spherical_num_elevation = 8  # Number of rays in vertical (elevation) direction
        # For spherical2 pattern (uniform)
        spherical2_num_points = 512     # Number of points for uniform spherical distribution
        spherical2_polar_axis = [0.0, 0.0, 1.0]  # Direction of polar axis
        # Cone pattern settings
        ray_angle = 30  # Angle of the cone in degrees
        # General settings
        num_rays = 96  # Total rays = spherical_num_azimuth * spherical_num_elevation
        max_distance = 10.0  # Maximum raycast distance
        attach_yaw_only = False  # Only consider yaw rotation for ray directions
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base

    class depth(LeggedRobotCfg.depth):
        camera_type = "Warp" # None, "IsaacGym", "Warp", "Fake"
        # BUG: if IsaacGym, the camera has no data when --headless

        position = [0.45, 0, 0.03]  # front camera
        angle = [30, 30]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (60, 30)
        resized = (56, 28)
        horizontal_fov = 100
        buffer_len = 2
        
        near_clip = 0
        far_clip = 10
        dis_noise = 0.0
        
        scale = 1
        invert = True

    class init_state(ElSpiderAirRoughTrainCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
    
    class commands(ElSpiderAirRoughTrainCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards(ElSpiderAirRoughTrainCfg.rewards):
        base_height_target = 0.35
        max_contact_force = 500.
        only_positive_rewards = True
        # Multi-stage rewards
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 5.0
        # Stage0-1: plane, Stage2: curriculum
        reward_min_stage = 2  # Start from 0
        reward_max_stage = 2

        class scales():

            # Tracking rewards
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # Base penalties
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = [-5.0, -5.0, 0.0]
            base_height = [-8.0, -8.0, 0.0]
            # DOF penalties
            torques = -0.00001
            dof_vel = -0.
            dof_acc = [-5e-8, -5e-8, -5e-8]
            dof_pos_limits = -1.0
            action_rate = [-0.001, -0.001, -0.002]
            # Feet penalties
            feet_slip = [-0.0, -0.4]  # Before feet_air_time
            feet_air_time = [0.8, 1.5]
            feet_stumble = [-1.0, -1.0, -2.0]
            feet_stumble_liftup = [1.0, 1.0, 2.0]
            feet_contact_forces = [0, 0, -0.05]  # Avoid jumping
            # Misc
            termination = -1.0
            collision = -1.
            stand_still = -0.
            # Gait
            async_gait_scheduler = [-0.2, -0.2, -0.1]
            gait_2_step = [-5.0, -5.0, -2.0]

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 0.3
            dof_nominal_pos = 0.2
            reward_foot_z_align = 0.0

    # class viewer:
    #     ref_env = 0
    #     pos = [30, 0, -10]  # [m]
    #     lookat = [0., 0, 0.]  # [m]
class ElSpiderAirRoughRaycastCfgPPO(ElSpiderAirRoughTrainCfgPPO):
    class runner(ElSpiderAirRoughTrainCfgPPO.runner):
        run_name = 'raycast512'
        experiment_name = 'rough_elspider_air'
        load_run = -1
        max_iterations = 5000  # number of policy updates

        multi_stage_rewards = True
