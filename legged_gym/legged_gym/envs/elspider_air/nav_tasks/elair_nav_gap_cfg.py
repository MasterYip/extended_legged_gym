"""Task-specific config for ElSpider Air - TimberPile navigation.

This file derives from elspider_air_nav_config.ElSpiderAirNavCfg and only
overrides the navigation (navi_opt) and terrain blocks as requested.
"""

from legged_gym.envs.elspider_air.batch_rollout.elspider_air_nav_config import (
    ElSpiderAirNavCfg,
    ElSpiderAirNavCfgPPO,
)

class ElAirNavGapCfg(ElSpiderAirNavCfg):
    """ElSpider Air configuration for a confined 'timber pile' navigation task.

    Only `terrain` and `navi_opt` are overridden here; all other settings are
    inherited from ElSpiderAirNavCfg.
    """

    class navi_opt(ElSpiderAirNavCfg.navi_opt):
        # ElSpider Air specific navigation settings with multiple start/goal positions
        start_pos = [
            [3.2, 3.2, 0.7],  # Environment 0
        ]
        
        # Corresponding goal positions
        goal_pos = [
            [9.6, 3.2, 0.7],  # Goal for Environment 0
        ]
        
        # Single orientation for all (can also be a list of orientations)
        start_quat = [
            [0.0, 0.0, 0.0, 1.0],
        ]
        
        tolerance_rad = 0.3           # Smaller tolerance for precision
        max_linear_vel = 1.5          # m/s
        max_angular_vel = 1.0         # rad/s (matching command ranges)
        kp_linear = 3.0               # Slightly higher gain
        kp_angular = 2.5              # Higher angular gain for better turning

    class terrain(ElSpiderAirNavCfg.terrain):
        mesh_type = 'confined_trimesh'
        # path to the terrain file
        measure_heights = False

        # Curriculum Settings
        curriculum = True
        max_init_terrain_level = 2
        terrain_length = 6.4
        terrain_width = 6.4
        num_rows = 2  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        difficulty_scale = 0.6
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.3, 0.2, 0.1] # FIXME: when a type=0, error might show up
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 0.0, 0.0, 1.0]
        spawn_area_size = 2.0
        # TerrainObj Settings
        use_terrain_obj = False  # use TerrainObj class to create terrain
        terrain_file = "resources/terrains/confined/confined_terrain.obj"
        # Origin generation method
        random_origins: bool = False  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        # origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        # origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        # height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height
        origins_x_range: list = [0.5, 1.5]  # min/max range for random x position
        origins_y_range: list = [-5, 1]  # min/max range for random y position
        height_clearance_factor: float = 1.5  # Required clearance as multiple of nominal_height


    class rewards(ElSpiderAirNavCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.34
        only_positive_rewards = False
        # No multi-stage rewards for flat environment
        multi_stage_rewards = True
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        class scales:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -0.5e-8
            # base_height = -8.0
            base_foot_height = -8.0
            feet_slip = [-0.0, -0.4]
            feet_air_time = 0.8
            collision = -0.8
            feet_stumble = -0.8
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.0
            dof_pos_limits = -1.0
            gait_2_step = -1.0

    class viewer(ElSpiderAirNavCfg.viewer):
        ref_env = None
        pos = [6.4, 6.4, 3.0]  # [m]
        lookat = [6.4, 3.2, 0.6]  # [m]
        render_rollouts = False


class ElAirNavGapCfgPPO(ElSpiderAirNavCfgPPO):
    class runner(ElSpiderAirNavCfgPPO.runner):
        experiment_name = 'elair_timberpile_nav'
        max_iterations = 2000

    # leave policy/algorithm defaults from parent
