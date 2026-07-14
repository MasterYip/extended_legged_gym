from legged_gym.envs.el_4090.spider_nomal.el4090_tripod2_low_config import (
    El4090Tripod2LowCfg,
    El4090Tripod2LowCfgPPO,
)

# LiDAR sensor constants
EA_SPHERICAL_AZIMUTH = 40
EA_SPHERICAL_ELEVATION = 25
EA_RAY_MAX_DISTANCE = 10.0
EA_PROPRIO_DIM = 66


class El4090EACfg(El4090Tripod2LowCfg):
    """EL_4090 Envelope Adaptive — LiDAR + external obstacle avoidance."""

    class env(El4090Tripod2LowCfg.env):
        num_observations = EA_PROPRIO_DIM
        num_privileged_obs = None
    
    class terrain(El4090Tripod2LowCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = False  #训练时True
        terrain_length = 16
        terrain_width = 16
        border_size = 5
        num_rows = 1  # number of terrain rows (levels) 训练时5
        num_cols = 2  # number of terrain cols (types) 训练时4
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
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
        pillar_spawn_radius = 7.5        #约束范围半径
        pillar_allow_height_variation = True

    class init_state(El4090Tripod2LowCfg.init_state):
        randomize_rot = False
        rot_randomization_range = [-3.14, 3.14]
        spawn_offset_range = 0.2

    # ── LiDAR 传感器配置（简化版，无 pd_risknet / cmd_safe） ──
    class raycaster:
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = EA_SPHERICAL_AZIMUTH
        spherical_num_elevation = EA_SPHERICAL_ELEVATION
        max_distance = EA_RAY_MAX_DISTANCE
        attach_yaw_only = False
        vertical_fov_deg_min = -2.0
        vertical_fov_deg_max = 57.0
        offset_pos = [0.0, 0.0, 0.25]
        sensor_offset_rpy = [0.0, 3.1416, 0.0]
        update_frequency_hz = 50.0

    # ── 包络参数上限 ──
    class envelope:
        x1_max = 1.5       # 前节点最大 x 坐标 (m)
        x3_max = -1.5      # 后节点最大 x 坐标 (m, 负=后方)
        front_rear_max = 0.8  # 前/后节点 l/r 最大延伸 (m)
        mid_max = 1.4         # 中节点 l/r 最大延伸 (m)
        z_top = 0.05          # 棱柱上层高度 (m)
        z_bottom = -0.25      # 棱柱下层高度 (m)
        margin_distance = 0.2   # outer hexagon offset distance (m)
        shrink_step = 0.03       # shrinkage per step (m)
        grow_step = 0.01        # recovery per step (m)

    # ── 避障速度规划 ──
    class avoidance:
        ground_threshold = 0.05   # world-frame Z ground filter (m)
        min_valid_dist = 0.15     # min valid hit distance (m)
        max_valid_dist = 10.0     # max valid (matches LiDAR range, m)
        ellipse_a = 0.6           # robot body ellipse semi-axis (forward, m)
        ellipse_b = 0.3           # robot body ellipse semi-axis (lateral, m)
        spline_smoothing = 0.8    # UnivariateSpline smoothing factor
        cmd_bias = 0.5            # bias toward cmd (m), added only to capped dirs
        cap_distance = 2.0       # distances > this are capped, then biased (m)
        n_azimuth = 40            # number of azimuth bins

    class sim(El4090Tripod2LowCfg.sim):
        class physx(El4090Tripod2LowCfg.sim.physx):
            max_gpu_contact_pairs = 2**23


class El4090EACfgPPO(El4090Tripod2LowCfgPPO):
    class runner(El4090Tripod2LowCfgPPO.runner):
        experiment_name = "el4090_ea"  # 复用 tripod2_low 训练好的模型
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
