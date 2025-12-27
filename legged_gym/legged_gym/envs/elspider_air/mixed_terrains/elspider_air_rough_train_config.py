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

#############################################################
# Multi Stage TrainingConfig for ElSpiderAir on Rough Terrain
#############################################################


# Basic
class ElSpiderAirRoughTrainCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 18
        num_observations = 66  + 187 # 253

    class terrain:
        use_terrain_obj = False  # use TerrainObj class to create terrain
        # path to the terrain file
        terrain_file = "/home/user/CodeSpace/Python/terrains/terrain.obj"

        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh, confined_trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 15  # [m] NOTE: margin of the terrain, too large will cause segmentation fault
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 1  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)
        difficulty_scale = 0.6
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.1, 0.3, 0.2, 0.4]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.6,
            "RM_HFE": 0.6,
            "RB_HFE": 0.6,
            "LF_HFE": 0.6,
            "LM_HFE": 0.6,
            "LB_HFE": 0.6,

            "RF_KFE": 0.6,
            "RM_KFE": 0.6,
            "RB_KFE": 0.6,
            "LF_KFE": 0.6,
            "LM_KFE": 0.6,
            "LB_KFE": 0.6,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}  # [N*m/rad]
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # Enable Network-0.5 | Disable Network-0.3

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini.urdf"
        name = "elspider_air"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "HIP", "trunk"]  # "SHANK" may collide with the ground through foot
        terminate_after_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 5.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # Reward V1
    class rewards(LeggedRobotCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.27
        only_positive_rewards = False
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait (with / w\o actuator net)
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        reward_min_stage = 1  # Start from 0
        reward_max_stage = 1

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = [-3.0, -1.0]
            torques = [-0.00002, -0.0001]
            dof_vel = [-0.0002, -0.001]
            dof_acc = [-3e-8, -2.5e-7]
            base_height = [-4.0, -2.0]
            # base_foot_height = -5.0
            feet_slip = [-0.0, -0.2]  # Before feet_air_time
            feet_air_time = [1.0, 1.0]
            collision = -1.
            feet_stumble = [-0.3, -1.0]
            feet_stumble_liftup = 5.0
            action_rate = [-0.003, -0.01]
            # stand_still = -0.4  # May affect spot turning
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.05, -0.1]
            
            # gait_scheduler = -18.0
            # async_gait_scheduler = -0.5  
            shank_perp2ground = [-0.3, -0.1] # Shanks to be perpendicular to the ground
            gait_2_step = [-2.0, -0.0]
    class domain_rand(LeggedRobotCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values
        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

# Stage0: Plane with Gait
class ElSpiderAirRoughStage0Cfg(ElSpiderAirRoughTrainCfg):
    class terrain(ElSpiderAirRoughTrainCfg.terrain):
        mesh_type = 'plane'  # none, plane, heightfield or trimesh, confined_trimesh

    class rewards(ElSpiderAirRoughTrainCfg.rewards):
        reward_min_stage = 0  # Start from 0

    class commands(ElSpiderAirRoughTrainCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.2, 1.2]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

# Stage1: Rough Terrain
class ElSpiderAirRoughStage1Cfg(ElSpiderAirRoughTrainCfg):
    class terrain(ElSpiderAirRoughTrainCfg.terrain):
        mesh_type = 'trimesh'  # none, plane, heightfield or trimesh, confined_trimesh

    class rewards(ElSpiderAirRoughTrainCfg.rewards):
        reward_min_stage = 1  # Start from 0
    
    class commands(ElSpiderAirRoughTrainCfg.commands):
        curriculum = False
        max_curriculum = 1.5
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.2, 1.2]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]


class ElSpiderAirRoughTrainCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_elair_multi_stage'
        load_run = -1
        max_iterations = 15000  # number of policy updates

        multi_stage_rewards = True
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # Symmetry augmentation configuration
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = "legged_gym.envs.elspider_air.elspider:get_elair_xysym_obs_act"
        