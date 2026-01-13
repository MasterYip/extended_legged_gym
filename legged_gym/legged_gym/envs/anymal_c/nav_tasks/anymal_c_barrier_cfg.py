"""Task-specific config for AnymalC - Barrier navigation.

This file derives from anymal_c_nav_config.AnymalCNavCfg and only
overrides the navigation (navi_opt) and terrain blocks as requested.
"""

from legged_gym.envs.anymal_c.batch_rollout.anymal_c_nav_config import (
    AnymalCNavCfg,
    AnymalCNavCfgPPO,
)
import numpy as np
from scipy.stats import qmc


def nav_start_goal_gen(num_pairs=50, 
                       start_area_x=[2.0, 4.0], 
                       start_area_y=[2.0, 4.0], 
                       start_z=0.5,
                       goal_margin=1.0,
                       max_distance=4.0,
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
    

class AnymalCNavBarrierCfg(AnymalCNavCfg):
    """AnymalC configuration for a confined 'barrier' navigation task.

    Only `terrain` and `navi_opt` are overridden here; all other settings are
    inherited from AnymalCNavCfg.
    """

    class navi_opt(AnymalCNavCfg.navi_opt):
        # AnymalC specific navigation settings with multiple start/goal positions
        start_pos, goal_pos, start_quat = nav_start_goal_gen(
            num_pairs=50, 
            start_area_x=[1.0, 7.0], 
            start_area_y=[1.0, 7.0], 
            start_z=0.5,
            goal_margin=1.2,
            max_distance=4.0,
            seed=42)
        
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
        num_rows = 1  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        difficulty_scale = 0.6
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.3, 0.2, 0.1]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 1.0, 0.0, 0.0]  # Only barriers
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
        multi_stage_rewards = False  # Keep it simple for navigation
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
            collision = -1.0  # Higher penalty for collisions in confined space
            feet_stumble = -0.8
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.

    class viewer(AnymalCNavCfg.viewer):
        ref_env = None
        pos = [-3, 4, 6.0]  # [m]
        lookat = [3.0, 4.0, 0.0]  # [m]
        render_rollouts = False


class AnymalCNavBarrierCfgPPO(AnymalCNavCfgPPO):
    class runner(AnymalCNavCfgPPO.runner):
        experiment_name = 'anymal_c_barrier_nav'
        max_iterations = 2000

    # leave policy/algorithm defaults from parent
