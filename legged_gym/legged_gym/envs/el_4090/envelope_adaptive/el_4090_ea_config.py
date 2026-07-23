from legged_gym.envs.el_4090.spider_nomal.el4090_tripod2_low_config import (
    El4090Tripod2LowCfg,
    El4090Tripod2LowCfgPPO,
)

# ── Airy LiDAR sensor constants ──
EA_AIRY_NUM_CHANNELS = 96                # 垂直通道数（激光线数）
EA_AIRY_HORIZONTAL_RES = 6.0             # 水平分辨率（度）
EA_AIRY_VERTICAL_FOV_MIN = 0.0           # 垂直 FOV 下界（度）
EA_AIRY_VERTICAL_FOV_MAX = 90.0          # 垂直 FOV 上界（度）
EA_RAY_MAX_DISTANCE = 60.0               # Airy 最大测距 (m)
EA_SPHERICAL_AZIMUTH = int(360.0 / EA_AIRY_HORIZONTAL_RES)   # 60（buffer 分配）
EA_SPHERICAL_ELEVATION = EA_AIRY_NUM_CHANNELS                # 96（buffer 分配）
EA_PROPRIO_DIM = 66
EA_CONDITION_DIM = 8
EA_NUM_OBS = EA_PROPRIO_DIM + EA_CONDITION_DIM  # 74


class El4090EACfg(El4090Tripod2LowCfg):
    """EL_4090 Envelope Adaptive — LiDAR + external obstacle avoidance.

    接口对齐底层 spider_envelop 训练配置：
    - num_observations=74, num_commands=12
    - control_type='P_LOWPASS', PD gains 130/4.0, action_scale=0.35
    - condition 8 维：5 长度 + 3 先验
    - 宽限位 URDF ([-3,3] joints)
    """

    class env(El4090Tripod2LowCfg.env):
        num_observations = EA_NUM_OBS
        num_privileged_obs = None

    class terrain(El4090Tripod2LowCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = False  # 训练时 True
        terrain_length = 16
        terrain_width = 16
        border_size = 5
        num_rows = 1  # number of terrain rows (levels) 训练时5
        num_cols = 2  # number of terrain cols (types) 训练时4
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
        difficulty_scale = 1.0

        # 柱子参数（pillar_field_terrain 已通过 getattr 读取）
        pillar_count_min = 12
        pillar_count_max = 12
        pillar_size_x_min = 0.5
        pillar_size_x_max = 4.0
        pillar_size_y_min = 0.5
        pillar_size_y_max = 4.0
        pillar_height_min = 1.00
        pillar_height_max = 2.00
        pillar_min_separation = 2.2
        pillar_center_clear_radius = 3.0
        pillar_spawn_radius = 7.5        # 约束范围半径
        pillar_allow_height_variation = True

        # 通道参数
        channel_width = 2.0
        wall_height = 1.5

    class commands(El4090Tripod2LowCfg.commands):
        num_commands = 4 + EA_CONDITION_DIM   # = 12
        condition_dim = EA_CONDITION_DIM
        curriculum = False
        resampling_time = 4.
        heading_command = False
        small_command_radio = True

        condition_names = [
            "front_width",
            "middle_width",
            "back_width",
            "forward_limit",
            "backward_limit",
            "morphology_front_prior",
            "morphology_middle_prior",
            "morphology_back_prior",
        ]
        morphology_prior_mode = "directional_ratio"
        morphology_prior_weights = {
            "front": {"lateral": 0.35, "longitudinal": 0.5},
            "middle": {"lateral": 1.0},
            "back": {"lateral": 0.35, "longitudinal": 0.5},
        }
        morphology_middle_front_follow_weight = 0.4

        class ranges(El4090Tripod2LowCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1., 1.]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            front_width = [0.3, 0.6]
            middle_width = [0.3, 0.7]
            back_width = [0.3, 0.6]
            forward_limit = [0.6, 0.9]
            backward_limit = [-0.9, -0.6]
            morphology_front_prior = [0.0, 1.0]
            morphology_middle_prior = [0.0, 1.0]
            morphology_back_prior = [0.0, 1.0]

    class control(El4090Tripod2LowCfg.control):
        control_type = 'P_LOWPASS'
        stiffness = {'HAA': 130., 'HFE': 130., 'KFE': 130.}
        damping = {'HAA': 4., 'HFE': 4., 'KFE': 4.}
        action_scale = 0.35
        decimation = 4
        default_dof_pos_filter_tau = 0.4
        default_dof_pos_filter_done_threshold = 0.02

    class init_state(El4090Tripod2LowCfg.init_state):
        pos = [0.0, 0.0, 0.52]
        randomize_rot = False
        rot_randomization_range = [-3.14, 3.14]
        spawn_offset_range = 0.2
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0.6, "RM_HFE": 0.6, "RB_HFE": 0.6,
            "LF_HFE": 0.6, "LM_HFE": 0.6, "LB_HFE": 0.6,
            "RF_KFE": -0.6, "RM_KFE": -0.6, "RB_KFE": -0.6,
            "LF_KFE": -0.6, "LM_KFE": -0.6, "LB_KFE": -0.6,
        }
        mammal_default_joint_angles = {
            "RF_HAA": -1.308, "RM_HAA": 1.308, "RB_HAA": 1.308,
            "LF_HAA": -1.308, "LM_HAA": 1.308, "LB_HAA": 1.308,
            "RF_HFE": 1., "RM_HFE": 1., "RB_HFE": 1.,
            "LF_HFE": 1., "LM_HFE": 1., "LB_HFE": 1.,
            "RF_KFE": -0.608, "RM_KFE": -0.608, "RB_KFE": -0.608,
            "LF_KFE": -0.608, "LM_KFE": -0.608, "LB_KFE": -0.608,
        }

    # ── 包络模块参数 ──
    # 收缩边界 = commands.ranges 的 5 长度范围(单一事实来源,不在此重复定义)
    class envelope:
        z_top = 0.15          # 棱柱上层高度 (m)
        z_bottom = -0.25      # 棱柱下层高度 (m)
        margin_distance = 0.25 # outer hexagon offset distance (m)
        hold_margin = 0.1       # shrink/hold 分界 (0~hold=shrink区, hold~margin=hold区)
        shrink_step = 0.03    # shrinkage per step (m)
        grow_step = 0.03      # recovery per step (m)
        grow_cooldown_frames = 5  # 连续无hit帧数阈值, 到达后开始扩张 (≈1 LiDAR cycle @10Hz)

    class rewards(El4090Tripod2LowCfg.rewards):
        # P5: reset 初始高度随形态先验插值(对齐底层 spider_envelop 训练环境)
        base_height_spider_target = 0.53
        base_height_mammal_target = 0.64
        reset_base_height_with_morphology = True

    class normalization(El4090Tripod2LowCfg.normalization):
        class obs_scales(El4090Tripod2LowCfg.normalization.obs_scales):
            embedded_state = 1.0

    class asset(El4090Tripod2LowCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_4090/urdf/el_4090_wide_limits.urdf"

    # ── 避障速度规划 ──
    class avoidance:
        enable = True              # False = 直接使用原指令, 不计算避障速度
        ground_threshold = 0.05   # world-frame Z ground filter (m)
        min_valid_dist = 0.15     # min valid hit distance (m)
        max_valid_dist = EA_RAY_MAX_DISTANCE
        ellipse_a = 0.6           # robot body ellipse semi-axis (forward, m)
        ellipse_b = 0.3           # robot body ellipse semi-axis (lateral, m)
        spline_smoothing = 0.8    # UnivariateSpline smoothing factor
        cmd_bias = 0.5            # bias toward cmd (m), added only to capped dirs
        cap_distance = 2.0        # distances > this are capped, then biased (m)
        n_azimuth = EA_SPHERICAL_AZIMUTH

    # ── Airy LiDAR 传感器配置 ──
    class raycaster:
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = EA_SPHERICAL_AZIMUTH    # 60
        spherical_num_elevation = EA_SPHERICAL_ELEVATION # 96
        max_distance = EA_RAY_MAX_DISTANCE               # 60.0
        attach_yaw_only = False
        vertical_fov_deg_min = EA_AIRY_VERTICAL_FOV_MIN  # 0.0
        vertical_fov_deg_max = EA_AIRY_VERTICAL_FOV_MAX  # 90.0
        offset_pos = [0.0, 0.0, -0.05]
        sensor_offset_rpy = [0.0, 0.0, 0.0]             # 面朝上方
        update_frequency_hz = 10.0                       # Airy 工作频率

    class sim(El4090Tripod2LowCfg.sim):
        class physx(El4090Tripod2LowCfg.sim.physx):
            max_gpu_contact_pairs = 2**23


class El4090EACfgPPO(El4090Tripod2LowCfgPPO):
    class policy(El4090Tripod2LowCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        init_noise_std = 0.3

    class runner(El4090Tripod2LowCfgPPO.runner):
        experiment_name = "el4090_ea"  # 复用底层训练好的模型(logs/el4090_ea/1)
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
