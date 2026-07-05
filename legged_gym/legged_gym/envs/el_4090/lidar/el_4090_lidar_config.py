from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)

# ── LiDAR perception constants (matching go2_cmd_safe) ──
PD_SPHERICAL_AZIMUTH = 40
PD_SPHERICAL_ELEVATION = 25
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION  # 1000
PD_PROXIMAL_POINTS = 256
PD_DISTAL_POINTS = 64
PD_DISTAL_HISTORY = 10
PD_SPLIT_THETA_DEG = 12.0
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
PD_N_SECTORS = 36
PD_RAY_MAX_DISTANCE = 10.0

# Height measurement grid
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
MEASURED_GRID_X_RANGE = [-1.8, 1.8]
MEASURED_GRID_Y_RANGE = [-1.2, 1.2]
PD_PRIV_HEIGHT_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT  # 187

# EL_4090 proprioceptive dimension: 3+3+3+3 + 18+18+18 = 66
PD_PROPRIO_DIM = 66


class El4090LidarCfg(El4090SpiderCfg):
    """Base config for EL_4090 with LiDAR perception.

    Adds spherical LiDAR sensor, sector-safety computation, and LiDAR-aware
    observation assembly on top of the standard EL_4090 setup.  Reward scales
    are intentionally left undefined here — subclasses (tripod, wave, mammal
    etc.) provide their own.

    LiDAR mounted at body centre, elevated +z, inverted (pitch=180°).
    """

    class env(El4090SpiderCfg.env):
        num_observations = PD_PROPRIO_DIM  # 66 (proprio only; LiDAR via lidar_points_base)
        num_privileged_obs = None           # no asymmetric critic; aux height via aux_obs_buf
        debug_viz = True

    class terrain(El4090SpiderCfg.terrain):
        measure_heights = True
        measured_grid_x_range = MEASURED_GRID_X_RANGE
        measured_grid_y_range = MEASURED_GRID_Y_RANGE
        measured_grid_x_count = MEASURED_GRID_X_COUNT
        measured_grid_y_count = MEASURED_GRID_Y_COUNT

    class init_state(El4090SpiderCfg.init_state):
        randomize_rot = False
        rot_randomization_range = [-3.14, 3.14]
        spawn_offset_range = 0.5

    # ── LiDAR perception parameters ──
    class pd_risknet:
        enabled = True
        num_lidar_points = PD_NUM_LIDAR_POINTS
        ray_max_distance = PD_RAY_MAX_DISTANCE
        n_sectors = PD_N_SECTORS
        split_theta_deg = PD_SPLIT_THETA_DEG

    # ── LiDAR sensor (raycaster) ──
    class raycaster:
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        max_distance = PD_RAY_MAX_DISTANCE
        attach_yaw_only = False
        vertical_fov_deg_min = -2.0
        vertical_fov_deg_max = 57.0
        # Sensor at body centre, elevated 0.25 m, inverted (pitch=180°)
        offset_pos = [0.0, 0.0, 0.25]
        sensor_offset_rpy = [0.0, 3.1416, 0.0]
        update_frequency_hz = 50.0

    # ── Sector-safety body dimensions ──
    class cmd_safe:
        body_semi_length = 0.25   # EL_4090 half-length, ~0.5 m body
        body_semi_width = 0.12    # EL_4090 half-width,  ~0.24 m body
        z_thresh_high = 0.15      # overhead filter threshold (body-frame z)
        d_safety = 0.10           # additional safety gap (m)
        d_safe_max = 1.0          # distance where safe = 1
        cmd_safe_sigma = 0.25     # gaussian kernel width
        dist_penalty_thresh = 1.0 # penalty activates below this (m)

    class domain_rand(El4090SpiderCfg.domain_rand):
        lidar_point_mask_ratio = 0.02
        lidar_point_mask_value_range = [0, 0.3]
        lidar_distance_noise_ratio = 0.02

    class commands(El4090SpiderCfg.commands):
        cmd_deadzone = 0.2  # threshold for zeroing small velocity commands


class El4090LidarCfgPPO(El4090SpiderCfgPPO):
    """Base PPO config for EL_4090 LiDAR perception tasks."""

    class policy(El4090SpiderCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        proximal_points = PD_PROXIMAL_POINTS
        distal_history_length = PD_DISTAL_HISTORY
        distal_points = PD_DISTAL_POINTS
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proprio_obs_dim = PD_PROPRIO_DIM
        privileged_height_dim = PD_PRIV_HEIGHT_DIM
        privileged_critic_dim = PD_PRIV_HEIGHT_DIM
        privileged_supervision_coef = 1.0
        gradient_checkpointing_proximal = True   # 近端,256步×187维
        gradient_checkpointing_distal = True     # 远端,1280步×64维

    class algorithm(El4090SpiderCfgPPO.algorithm):
        use_amp = True
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 1.0
            data_augmentation_func = "legged_gym.envs.el_4090.lidar.el_4090_lidar_symmetry:get_el4090_lidar_xsym_obs_act"
            # Dimension parameters injected into the symmetry function by the runner.
            # sensor_quat and sensor_trans are NOT declared here — the runner
            # extracts them from the environment at initialisation time.
            symmetry_kwargs = dict(
                proprio_dim=PD_PROPRIO_DIM,           # 66
                proximal_points=PD_PROXIMAL_POINTS,    # 256
                distal_history_points=PD_DISTAL_HISTORY * PD_DISTAL_POINTS,  # 10 * 64 = 640
                num_dof=18,                            # EL_4090: 6 legs × 3 joints
                height_grid_x_count=MEASURED_GRID_X_COUNT,  # 17
                height_grid_y_count=MEASURED_GRID_Y_COUNT,  # 11
            )

        aux_loss_coef = 1.0

    class runner(El4090SpiderCfgPPO.runner):
        policy_class_name = "LidarPDActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 4000
        save_interval = 50
        experiment_name = "el_4090_lidar"
