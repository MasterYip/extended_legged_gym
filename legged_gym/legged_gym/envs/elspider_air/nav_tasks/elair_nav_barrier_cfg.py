"""Task-specific config for ElSpider Air - Barrier navigation.

This file derives from elspider_air_nav_config.ElSpiderAirNavCfg and only
overrides the navigation (navi_opt) and terrain blocks as requested.
"""

from legged_gym.envs.elspider_air.batch_rollout.elspider_air_nav_config import (
    ElSpiderAirNavCfg,
    ElSpiderAirNavCfgPPO,
)
import numpy as np
from scipy.stats import qmc


def nav_start_goal_gen(num_pairs=50, 
                       start_area_x=[2.0, 4.0], 
                       start_area_y=[2.0, 4.0], 
                       start_z=0.32,
                       goal_margin=1.0,
                       max_distance=3.0,
                       seed=42):
    """Generate start/goal pairs using Latin Hypercube Sampling for better space coverage.
    
    Args:
        num_pairs: Number of start/goal pairs to generate
        start_area_x: [min, max] x coordinates for start positions
        start_area_y: [min, max] y coordinates for start positions
        start_z: z coordinate for all positions
        goal_margin: Minimum margin between start area and goals
        max_distance: Maximum allowed distance between start and goal positions
        seed: Random seed for reproducibility
    """
    
    # Use Latin Hypercube Sampling for better space coverage
    sampler = qmc.LatinHypercube(d=5, seed=seed)  # 5D: start_x, start_y, goal_x, goal_y, theta
    
    start_pos = []
    goal_pos = []
    start_quat = []
    
    # Generate extra samples to account for rejections due to distance constraint
    max_attempts = num_pairs * 3  # Generate 3x samples to handle rejections
    samples = sampler.random(n=max_attempts)
    
    sample_idx = 0
    valid_pairs = 0
    
    while valid_pairs < num_pairs and sample_idx < max_attempts:
        sample = samples[sample_idx]
        sample_idx += 1
        
        # Start position: map [0,1] to start area
        sx = float(start_area_x[0] + sample[0] * (start_area_x[1] - start_area_x[0]))
        sy = float(start_area_y[0] + sample[1] * (start_area_y[1] - start_area_y[0]))

        # Goal position: map [0,1] to expanded area, then check margin
        expanded_min_x = start_area_x[0] - 2.0 - goal_margin
        expanded_max_x = start_area_x[1] + 2.0 + goal_margin
        expanded_min_y = start_area_y[0] - 2.0 - goal_margin
        expanded_max_y = start_area_y[1] + 2.0 + goal_margin
        
        gx = float(expanded_min_x + sample[2] * (expanded_max_x - expanded_min_x))
        gy = float(expanded_min_y + sample[3] * (expanded_max_y - expanded_min_y))
        
        # If goal is inside the forbidden zone, push it out
        if (start_area_x[0] - goal_margin <= gx <= start_area_x[1] + goal_margin and
            start_area_y[0] - goal_margin <= gy <= start_area_y[1] + goal_margin):
            # Push to nearest edge
            center_x = (start_area_x[0] + start_area_x[1]) / 2
            center_y = (start_area_y[0] + start_area_y[1]) / 2
            
            if abs(gx - center_x) > abs(gy - center_y):
                # Push in x direction
                if gx < center_x:
                    gx = start_area_x[0] - goal_margin - 0.5
                else:
                    gx = start_area_x[1] + goal_margin + 0.5
            else:
                # Push in y direction
                if gy < center_y:
                    gy = start_area_y[0] - goal_margin - 0.5
                else:
                    gy = start_area_y[1] + goal_margin + 0.5

        # Check distance constraint
        distance = np.sqrt((gx - sx)**2 + (gy - sy)**2)
        if distance > max_distance:
            continue  # Skip this sample and try the next one
        
        # Valid pair found - add to lists
        start_pos.append([sx, sy, start_z])
        goal_pos.append([gx, gy, start_z])

        # Random yaw-only quaternion using LHS sample
        theta = float(sample[4] * 2.0 * np.pi)  # Map [0,1] to [0,2π]
        start_quat.append([0.0, 0.0, float(np.sin(theta / 2.0)), float(np.cos(theta / 2.0))])
        
        valid_pairs += 1
    
    # If we couldn't generate enough valid pairs, fill remaining with closest valid ones
    if valid_pairs < num_pairs:
        print(f"Warning: Only generated {valid_pairs}/{num_pairs} valid start/goal pairs within max_distance={max_distance}")
        # Duplicate the last valid pair to fill remaining slots
        while len(start_pos) < num_pairs:
            if start_pos:  # Only if we have at least one valid pair
                start_pos.append(start_pos[-1])
                goal_pos.append(goal_pos[-1])
                start_quat.append(start_quat[-1])
            else:
                # Fallback: create a simple valid pair
                start_pos.append([start_area_x[0], start_area_y[0], start_z])
                goal_pos.append([start_area_x[1], start_area_y[1], start_z])
                start_quat.append([0.0, 0.0, 0.0, 1.0])
        
    return start_pos, goal_pos, start_quat
    
class ElAirNavBarrierCfg(ElSpiderAirNavCfg):
    """ElSpider Air configuration for a confined 'barrier' navigation task.

    Only `terrain` and `navi_opt` are overridden here; all other settings are
    inherited from ElSpiderAirNavCfg.
    """

    class navi_opt(ElSpiderAirNavCfg.navi_opt):
        # ElSpider Air specific navigation settings with multiple start/goal positions
        start_pos, goal_pos, start_quat = nav_start_goal_gen(
            num_pairs = 50, 
            start_area_x = [1.0, 7.0], 
            start_area_y = [1.0, 7.0], 
            start_z = 0.32,
            goal_margin = 1.0,
            max_distance = 3.0,
            seed=42)
        
        tolerance_rad = 0.3           # Smaller tolerance for precision
        max_linear_vel = 1.0          # m/s
        max_angular_vel = 0.8         # rad/s (matching command ranges)
        kp_linear = 3.0               # Slightly higher gain
        kp_angular = 2.5              # Higher angular gain for better turning

    class terrain(ElSpiderAirNavCfg.terrain):
        mesh_type = 'confined_trimesh'
        # path to the terrain file
        measure_heights = False

        # Curriculum Settings
        curriculum = True
        max_init_terrain_level = 2
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        difficulty_scale = 0.6
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.3, 0.2, 0.1] # FIXME: when a type=0, error might show up
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 1.0, 0.0, 0.0]
        spawn_area_size = 6.0
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
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -0.5e-8
            base_height = -8.0
            feet_slip = [-0.0, -0.4]
            feet_air_time = 0.8
            collision = -0.5
            feet_stumble = -0.4
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.0
            dof_pos_limits = -1.0
            gait_2_step = -1.0

    class viewer(ElSpiderAirNavCfg.viewer):
        ref_env = None
        pos = [-3, 4, 6.0]  # [m]
        lookat = [3.0, 4.0, 0.0]  # [m]
        render_rollouts = False


class ElAirNavBarrierCfgPPO(ElSpiderAirNavCfgPPO):
    class runner(ElSpiderAirNavCfgPPO.runner):
        experiment_name = 'elair_barrier_nav'
        max_iterations = 2000

    # leave policy/algorithm defaults from parent
