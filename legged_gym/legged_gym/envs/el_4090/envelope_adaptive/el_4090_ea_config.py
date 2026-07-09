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
        x1_max = 2.0       # 前节点最大 x 坐标 (m)
        x3_max = -2.0      # 后节点最大 x 坐标 (m, 负=后方)
        front_rear_max = 2.0  # 前/后节点 l/r 最大延伸 (m)
        mid_max = 3.0         # 中节点 l/r 最大延伸 (m)
        z_top = 0.25          # 棱柱上层高度 (m)
        z_bottom = -0.05      # 棱柱下层高度 (m)

    class sim(El4090Tripod2LowCfg.sim):
        class physx(El4090Tripod2LowCfg.sim.physx):
            max_gpu_contact_pairs = 2**24


class El4090EACfgPPO(El4090Tripod2LowCfgPPO):
    class runner(El4090Tripod2LowCfgPPO.runner):
        experiment_name = "el4090_tripod2_low"  # 复用 tripod2_low 训练好的模型
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
