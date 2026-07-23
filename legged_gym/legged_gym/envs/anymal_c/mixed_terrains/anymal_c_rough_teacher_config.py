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

class AnymalCRoughTeacherCfg( LeggedRobotCfg ):

    class env:
        num_envs = 4096
        num_observations = 235
        # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). 
        # if None privileged_obs will be deemed as obs
        num_privileged_obs = None  
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain:
        use_terrain_obj = False  # use TerrainObj class to create terrain
        # path to the terrain file (rel path under LEGGED_GYM_ROOT_DIR / abs path)
        terrain_file = "/home/user/CodeSpace/Python/terrains/terrain.obj"

        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
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
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 5.
        terrain_width = 5.
        num_rows = 8  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": -0.0,
            "RH_HAA": -0.0,

            "LF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RF_HFE": 0.4,
            "RH_HFE": -0.4,

            "LF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RF_KFE": -0.8,
            "RH_KFE": 0.8,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

class AnymalCRoughTeacherCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO' # PPO or Distillation
        run_name = ''
        experiment_name = 'rough_anymal_c_teacher'
        load_run = -1
