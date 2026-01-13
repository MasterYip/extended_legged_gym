"""Task-specific config for AnymalC - TimberPile navigation.

This file derives from anymal_c_nav_config.AnymalCNavCfg and only
overrides the navigation (navi_opt) and terrain blocks as requested.
"""

from legged_gym.envs.anymal_c.batch_rollout.anymal_c_nav_config import (
    AnymalCNavCfg,
    AnymalCNavCfgPPO,
)

class AnymalCNavTimberPileCfg(AnymalCNavCfg):
    """AnymalC configuration for a confined 'timber pile' navigation task.

    Only `terrain` and `navi_opt` are overridden here; all other settings are
    inherited from AnymalCNavCfg.
    """

    class navi_opt(AnymalCNavCfg.navi_opt):
        # AnymalC specific navigation settings with multiple start/goal positions
        start_pos = [
            [2.5, 2.5, 0.5],  # Environment 0
            [3.5, 2.5, 0.5],  # Environment 1
            [2.5, 3.5, 0.5],  # Environment 2
            [3.5, 3.5, 0.5],  # Environment 3
        ]
        
        # Corresponding goal positions
        goal_pos = [
            [3.0, 0.0, 0.5],  # Goal for Environment 0
            [6.0, 3.0, 0.5],  # Goal for Environment 1  
            [0.0, 3.0, 0.5],  # Goal for Environment 2
            [3.0, 6.0, 0.5],  # Goal for Environment 3
        ]
        
        # Single orientation for all (can also be a list of orientations)
        start_quat = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.707, 0.707],
            [0.0, 0.0, -0.707, 0.707],
            [0.0, 0.0, -1.0, 0.0],
        ]
        
        tolerance_rad = 0.4           # Tolerance for goal reaching
        max_linear_vel = 1.2          # m/s (slightly conservative for AnymalC)
        max_angular_vel = 1.0         # rad/s (matching command ranges)
        kp_linear = 2.5               # Linear velocity gain
        kp_angular = 2.0              # Angular velocity gain

    class terrain(AnymalCNavCfg.terrain):
        mesh_type = 'confined_trimesh'
        # path to the terrain file
        terrain_file = "/home/chen/extended_legged_gym/legged_gym/resources/terrains/confined/confined_terrain.obj"
        measure_heights = False

        # Curriculum Settings
        curriculum = True
        max_init_terrain_level = 2
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 2  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        difficulty_scale = 0.6
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.3, 0.2, 0.1]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 0.0, 1.0, 0.0]  # Only timber piles
        spawn_area_size = 6.0
        
        # Origin generation method
        random_origins: bool = False  # Use fixed origins for navigation task
        origin_generation_max_attempts: int = 10000  
        origins_x_range: list = [0.5, 1.5]  # min/max range for random x position
        origins_y_range: list = [-5, 1]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    class rewards(AnymalCNavCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.5
        only_positive_rewards = False
        # Multi-stage rewards
        multi_stage_rewards = True  # Enable for timber pile navigation
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        class scales():
            termination = -0.0
            tracking_lin_vel = 2.5  # Slightly higher for navigation focus
            tracking_ang_vel = 0.8
            lin_vel_z = -1.0
            ang_vel_xy = -0.5
            orientation = -4.0
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_foot_height = -10
            feet_air_time = 0.4
            collision = -0.8  # Higher penalty for collisions in confined space
            feet_stumble = -0.8
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.
            dof_pos_limits = -1.0

    class viewer(AnymalCNavCfg.viewer):
        ref_env = None
        pos = [-3, 4, 6.0]  # [m] - bird's eye view for navigation
        lookat = [3.0, 4.0, 0.]  # [m] - look at center of arena
        render_rollouts = False


class AnymalCNavTimberPileCfgPPO(AnymalCNavCfgPPO):
    class runner(AnymalCNavCfgPPO.runner):
        experiment_name = 'anymal_c_timberpile_nav'
        max_iterations = 2000

    # leave policy/algorithm defaults from parent
