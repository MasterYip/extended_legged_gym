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

from legged_gym.envs.elspider_air.mixed_terrains.elspider_air_rough_config import ElSpiderAirRoughCfg, ElSpiderAirRoughCfgPPO
from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import El4090SpiderCfg,El4090SpiderCfgPPO


class El4090SafeCfg(ElSpiderAirRoughCfg):
    class env(ElSpiderAirRoughCfg.env):
        num_envs = 4096
        num_observations = 143
        num_height_start_idx = 66
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 18
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        # Debug settings
        debug_mode = False  # Enable debug output
        debug_interval = 100  # Print debug info every N steps
        debug_env_id = 0  # Which environment to debug (0-based index)
    
    

    class terrain(ElSpiderAirRoughCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
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
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 10.
        terrain_width = 10.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        difficulty_scale = 0.0
        terrain_proportions = [0., 1., 0., 0., 0.]
        # terrain_proportions = [0.2, 0.15, 0.15, 0.2, 0.3]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class control(ElSpiderAirRoughCfg.control):
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 100., 
                     'HFE': 100., 
                     'KFE': 100.}  # [N*m/rad]
        damping = {'HAA': 1.2, 
                   'HFE': 1.2, 
                   'KFE': 1.2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class asset(ElSpiderAirRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_4090/urdf/el_4090.urdf"
        name = "el_4090"
        foot_name = "FOOT"
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        shoulder_name = "shoulder"
        penalize_contacts_on = ["BASE","SHANK","THIGH"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class init_state(ElSpiderAirRoughCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.0,
            "RM_HFE": 0.0,
            "RB_HFE": 0.0,
            "LF_HFE": 0.0,
            "LM_HFE": 0.0,
            "LB_HFE": 0.0,

            "RF_KFE": 0.0,
            "RM_KFE": 0.0,
            "RB_KFE": 0.0,
            "LF_KFE": 0.0,
            "LM_KFE": 0.0,
            "LB_KFE": 0.0,
        }

    ## Rewards V1 (normal dof_acc)
    class rewards(ElSpiderAirRoughCfg.rewards):
        max_contact_force = 250.
        base_height_target = 0.47
        only_positive_rewards = False
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 5
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales:

            termination = -0.0

            tracking_lin_vel = 8
            tracking_ang_vel = 5.5
            # lateral_lin_vel_y = -1

            lin_vel_z = -2
            ang_vel_xy = -1
            orientation = -5
            torques = -0.0001
            dof_vel = -0.0001
            dof_acc = -1e-7
            base_height = -50
            feet_slip = -0.05 
            feet_air_time = 1.5
            collision = -1.
            feet_stumble = -1
            action_rate = -0.03
            stand_still = -1.5  
            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01
            feet_contact_forces = -0.04
            # stand_on_six_legs = -1
            shank_vertical = -4
            feet_async = -3
            feet_sync = -3


    class commands(ElSpiderAirRoughCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = True

        class ranges(ElSpiderAirRoughCfg.commands.ranges):
            lin_vel_x = [-4.0, 4.0]  # min max [m/s]
            lin_vel_y = [-1.5, 1.5]   # min max [m/s]
            ang_vel_yaw = [-2.0, 2.0]    # min max [rad/s]
            heading = [-1, 1]

    class domain_rand(ElSpiderAirRoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-10., 10.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.

    class noise(ElSpiderAirRoughCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.0
            lin_vel = 1.2
            ang_vel = 1.2
            gravity = 1.2
            height_measurements = 0.1

    class safety:
        # 核心开关
        enable_atacom = True              # 是否启用 ATACOM 安全层
        clip_nominal_actions = True       # 是否在送入 ATACOM 前裁剪名义动作
        warmup_steps = 0                  # 前 N 步跳过 ATACOM（用于调试）

        # 算法超参数
        lambda_retract = 5.0              # 收缩增益 λ：控制向约束流形收缩的速率
        beta = 2.0                        # 松弛变量动力学系数
        dt = 0.005                        # 控制步长（s），建议与仿真 dt 保持一致

        # 关节限位（列表长度须为 18）
        q_max = [1.57] * 18               # 关节位置上限（rad）
        q_min = [-1.57] * 18              # 关节位置下限（rad）
        dq_max = [15] * 18                # 关节速度上限（rad/s）
        tau_max = [76] * 18               # 关节力矩上限（N·m）
        # q_max = [3.0] * 18               # 关节位置上限（rad）
        # q_min = [-3.0] * 18              # 关节位置下限（rad）
        # dq_max = [20] * 18               # 关节速度上限（rad/s）
        # tau_max = [80] * 18               # 关节力矩上限（N·m）

        # 机身限位
        phi_max = [0.14, 0.14, 3.14]      # roll, pitch, yaw 上限（rad）
        z_min = 0.2                       # 机身高度下限（m）
        z_max = 0.8                       # 机身高度上限（m）

        # 日志配置
        log_info = False                  # 是否将 forward() 返回的 info 聚合为标量（触发 GPU 同步）
        record_violation = True           # 训练时记录约束违反信息到 safe/data/*.csv
        record_violation_interval = 1     # 每 N 个环境步记录一次
        record_violation_detail = True    # 是否记录“哪些约束违反了”（索引/名称/top-k）
        record_violation_topk = 5         # 每步记录违反量最大的前 K 个约束

        # 调试日志配置
        debug_mode = False                # 是否启用调试日志
        debug_level = 'basic'             # 日志级别: basic/verbose/debug
        debug_interval = 100              # 日志输出间隔（步数）

class El4090SafeCfgPPO( El4090SpiderCfgPPO ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.3
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm(ElSpiderAirRoughCfgPPO.algorithm):
        # Symmetry augmentation configuration
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = "legged_gym.envs.el_4090.safe.el_4090_safe_symmetry:get_el4090_safe_xsym_obs_act"
        
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'el_4090_safe'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt