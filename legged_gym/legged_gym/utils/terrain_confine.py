import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# Configuration constants for confined terrain
SPAWN_AREA_SIZE = 2.0  # meters - size of central robot spawn area (configurable)
ENABLE_CEILING = False
GLOBAL_NOISE = 0.01  # Default global noise amplitude

def convert_2layer_heightfield_to_trimesh(ground_height_field, ceiling_height_field, 
                                          horizontal_scale, vertical_scale, 
                                          slope_threshold=None, enable_ceiling=ENABLE_CEILING,
                                          global_noise=GLOBAL_NOISE):
    """
    Convert two heightfield arrays (ground and ceiling) to a triangle mesh represented by vertices and triangles.
    This creates a confined environment with both floor and ceiling surfaces.
    
    Parameters:
        ground_height_field (np.array): input ground heightfield
        ceiling_height_field (np.array): input ceiling heightfield  
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
        enable_ceiling (bool): whether to include ceiling triangles in the mesh (default: False)
        global_noise (float): amplitude of noise to apply to the entire terrain map (default: 0.0)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    ground_hf = ground_height_field.copy()  # Make a copy to avoid modifying original
    ceiling_hf = ceiling_height_field.copy()  # Make a copy to avoid modifying original
    num_rows = ground_hf.shape[0]
    num_cols = ground_hf.shape[1]
    
    assert ground_hf.shape == ceiling_hf.shape, "Ground and ceiling heightfields must have same dimensions"

    # Apply global noise if specified
    if global_noise > 0:
        # Generate random noise for the entire terrain
        noise_ground = np.random.uniform(-global_noise, global_noise, (num_rows, num_cols))
        noise_ceiling = np.random.uniform(-global_noise, global_noise, (num_rows, num_cols))
        
        # Add noise to height fields (scaled by vertical_scale)
        ground_hf = ground_hf + (noise_ground / vertical_scale).astype(ground_hf.dtype)
        ceiling_hf = ceiling_hf + (noise_ceiling / vertical_scale).astype(ceiling_hf.dtype)

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        
        # Apply slope correction to ground
        move_x_ground = np.zeros((num_rows, num_cols))
        move_y_ground = np.zeros((num_rows, num_cols))
        move_corners_ground = np.zeros((num_rows, num_cols))
        move_x_ground[:num_rows-1, :] += (ground_hf[1:num_rows, :] - ground_hf[:num_rows-1, :] > slope_threshold)
        move_x_ground[1:num_rows, :] -= (ground_hf[:num_rows-1, :] - ground_hf[1:num_rows, :] > slope_threshold)
        move_y_ground[:, :num_cols-1] += (ground_hf[:, 1:num_cols] - ground_hf[:, :num_cols-1] > slope_threshold)
        move_y_ground[:, 1:num_cols] -= (ground_hf[:, :num_cols-1] - ground_hf[:, 1:num_cols] > slope_threshold)
        move_corners_ground[:num_rows-1, :num_cols-1] += (ground_hf[1:num_rows, 1:num_cols] - ground_hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners_ground[1:num_rows, 1:num_cols] -= (ground_hf[:num_rows-1, :num_cols-1] - ground_hf[1:num_rows, 1:num_cols] > slope_threshold)
        
        # Apply slope correction to ceiling only if enabled
        if enable_ceiling:
            move_x_ceiling = np.zeros((num_rows, num_cols))
            move_y_ceiling = np.zeros((num_rows, num_cols))
            move_corners_ceiling = np.zeros((num_rows, num_cols))
            move_x_ceiling[:num_rows-1, :] += (ceiling_hf[1:num_rows, :] - ceiling_hf[:num_rows-1, :] > slope_threshold)
            move_x_ceiling[1:num_rows, :] -= (ceiling_hf[:num_rows-1, :] - ceiling_hf[1:num_rows, :] > slope_threshold)
            move_y_ceiling[:, :num_cols-1] += (ceiling_hf[:, 1:num_cols] - ceiling_hf[:, :num_cols-1] > slope_threshold)
            move_y_ceiling[:, 1:num_cols] -= (ceiling_hf[:, :num_cols-1] - ceiling_hf[:, 1:num_cols] > slope_threshold)
            move_corners_ceiling[:num_rows-1, :num_cols-1] += (ceiling_hf[1:num_rows, 1:num_cols] - ceiling_hf[:num_rows-1, :num_cols-1] > slope_threshold)
            move_corners_ceiling[1:num_rows, 1:num_cols] -= (ceiling_hf[:num_rows-1, :num_cols-1] - ceiling_hf[1:num_rows, 1:num_cols] > slope_threshold)
        
        # Create separate coordinate grids for ground and ceiling
        xx_ground = xx + (move_x_ground + move_corners_ground*(move_x_ground == 0)) * horizontal_scale
        yy_ground = yy + (move_y_ground + move_corners_ground*(move_y_ground == 0)) * horizontal_scale
        if enable_ceiling:
            xx_ceiling = xx + (move_x_ceiling + move_corners_ceiling*(move_x_ceiling == 0)) * horizontal_scale
            yy_ceiling = yy + (move_y_ceiling + move_corners_ceiling*(move_y_ceiling == 0)) * horizontal_scale
        else:
            xx_ceiling = xx
            yy_ceiling = yy
    else:
        xx_ground = xx_ceiling = xx
        yy_ground = yy_ceiling = yy

    # Create triangle mesh vertices for ground and optionally ceiling
    total_vertices = num_rows * num_cols * (2 if enable_ceiling else 1)
    vertices = np.zeros((total_vertices, 3), dtype=np.float32)
    
    # Ground vertices (first half)
    vertices[:num_rows*num_cols, 0] = xx_ground.flatten()
    vertices[:num_rows*num_cols, 1] = yy_ground.flatten()
    vertices[:num_rows*num_cols, 2] = ground_hf.flatten() * vertical_scale
    
    # Ceiling vertices (second half) - only if enabled
    if enable_ceiling:
        vertices[num_rows*num_cols:, 0] = xx_ceiling.flatten()
        vertices[num_rows*num_cols:, 1] = yy_ceiling.flatten()
        vertices[num_rows*num_cols:, 2] = ceiling_hf.flatten() * vertical_scale

    # Create triangles for ground and optionally ceiling
    total_triangles = 2 * (num_rows-1) * (num_cols-1) * (2 if enable_ceiling else 1)
    triangles = -np.ones((total_triangles, 3), dtype=np.uint32)
    
    # Ground triangles (normal pointing up)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    # Ceiling triangles (normal pointing down) - only if enabled
    if enable_ceiling:
        offset = num_rows * num_cols
        tri_offset = 2 * (num_rows-1) * (num_cols-1)
        for i in range(num_rows - 1):
            ind0 = np.arange(0, num_cols-1) + i*num_cols + offset
            ind1 = ind0 + 1
            ind2 = ind0 + num_cols
            ind3 = ind2 + 1
            start = tri_offset + 2*i*(num_cols-1)
            stop = start + 2*(num_cols-1)
            # Reverse winding order for ceiling (normal pointing down)
            triangles[start:stop:2, 0] = ind0
            triangles[start:stop:2, 1] = ind1
            triangles[start:stop:2, 2] = ind3
            triangles[start+1:stop:2, 0] = ind0
            triangles[start+1:stop:2, 1] = ind3
            triangles[start+1:stop:2, 2] = ind2

    return vertices, triangles


def tunnel_terrain(terrain_ground, terrain_ceiling, tunnel_width=1.0, tunnel_height=2.0, wall_thickness=0.5):
    """
    Generate a tunnel terrain with 4 tunnels extending from central spawn area
    
    Parameters:
        terrain_ground: ground SubTerrain object
        terrain_ceiling: ceiling SubTerrain object  
        tunnel_width: width of the tunnel opening [meters]
        tunnel_height: height of the tunnel [meters]
        wall_thickness: thickness of tunnel walls [meters]
    """
    # Convert to discrete units
    tunnel_width_px = int(tunnel_width / terrain_ground.horizontal_scale)
    tunnel_height_units = int(tunnel_height / terrain_ground.vertical_scale)
    wall_thickness_px = int(wall_thickness / terrain_ground.horizontal_scale)
    
    center_x = terrain_ground.width // 2
    center_y = terrain_ground.length // 2
    
    # Create central spawn area
    spawn_area_size_px = int(SPAWN_AREA_SIZE / terrain_ground.horizontal_scale)
    spawn_half_size = spawn_area_size_px // 2
    spawn_x1 = center_x - spawn_half_size
    spawn_x2 = center_x + spawn_half_size
    spawn_y1 = center_y - spawn_half_size
    spawn_y2 = center_y + spawn_half_size
    
    # Keep spawn area flat (ground level 0)
    terrain_ground.ground_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = 0
    terrain_ceiling.ceiling_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = tunnel_height_units
    
    # Create 4 tunnels extending from spawn area in cardinal directions
    tunnel_half_width = tunnel_width_px // 2
    
    # North tunnel (positive Y direction)
    if spawn_y2 < terrain_ground.length:
        north_y1 = max(center_y - tunnel_half_width, 0)
        north_y2 = min(center_y + tunnel_half_width, terrain_ground.length)
        terrain_ground.ground_height_field_raw[spawn_y2:, north_y1:north_y2] = -int(0.1 / terrain_ground.vertical_scale)
        terrain_ceiling.ceiling_height_field_raw[spawn_y2:, north_y1:north_y2] = int(1.2 / terrain_ceiling.vertical_scale)
    
    # South tunnel (negative Y direction)
    if spawn_y1 > 0:
        south_y1 = max(center_y - tunnel_half_width, 0)
        south_y2 = min(center_y + tunnel_half_width, terrain_ground.length)
        terrain_ground.ground_height_field_raw[:spawn_y1, south_y1:south_y2] = -int(0.1 / terrain_ground.vertical_scale)
        terrain_ceiling.ceiling_height_field_raw[:spawn_y1, south_y1:south_y2] = int(1.2 / terrain_ceiling.vertical_scale)
    
    # East tunnel (positive X direction)
    if spawn_x2 < terrain_ground.width:
        east_x1 = max(center_x - tunnel_half_width, 0)
        east_x2 = min(center_x + tunnel_half_width, terrain_ground.width)
        terrain_ground.ground_height_field_raw[east_x1:east_x2, spawn_x2:] = -int(0.1 / terrain_ground.vertical_scale)
        terrain_ceiling.ceiling_height_field_raw[east_x1:east_x2, spawn_x2:] = int(1.2 / terrain_ceiling.vertical_scale)
    
    # West tunnel (negative X direction)
    if spawn_x1 > 0:
        west_x1 = max(center_x - tunnel_half_width, 0)
        west_x2 = min(center_x + tunnel_half_width, terrain_ground.width)
        terrain_ground.ground_height_field_raw[west_x1:west_x2, :spawn_x1] = -int(0.1 / terrain_ground.vertical_scale)
        terrain_ceiling.ceiling_height_field_raw[west_x1:west_x2, :spawn_x1] = int(1.2 / terrain_ceiling.vertical_scale)
    
    # Set default ceiling height for non-tunnel areas
    default_ceiling = int(3.0 / terrain_ceiling.vertical_scale)
    mask = terrain_ceiling.ceiling_height_field_raw == int(3.0 / terrain_ceiling.vertical_scale)  # Default areas
    terrain_ceiling.ceiling_height_field_raw[mask] = default_ceiling

    return terrain_ground, terrain_ceiling


def barrier_terrain(terrain_ground, terrain_ceiling, barrier_width=0.35, barrier_height=0.2, gap_height=0.8):
    """
    Generate square barrier around central spawn area
    
    Parameters:
        terrain_ground: ground SubTerrain object
        terrain_ceiling: ceiling SubTerrain object
        barrier_width: width of barriers [meters]
        barrier_height: height of ground barriers [meters] 
        gap_height: height of the gap between barriers [meters]
    """
    # Convert to discrete units
    barrier_width_px = int(barrier_width / terrain_ground.horizontal_scale)
    barrier_height_units = int(barrier_height / terrain_ground.vertical_scale)
    gap_height_units = int(gap_height / terrain_ceiling.vertical_scale)
    
    center_x = terrain_ground.width // 2
    center_y = terrain_ground.length // 2
    
    # Create central spawn area
    spawn_area_size_px = int(SPAWN_AREA_SIZE / terrain_ground.horizontal_scale)
    spawn_half_size = spawn_area_size_px // 2
    spawn_x1 = center_x - spawn_half_size
    spawn_x2 = center_x + spawn_half_size
    spawn_y1 = center_y - spawn_half_size
    spawn_y2 = center_y + spawn_half_size
    
    # Keep spawn area flat (ground level 0)
    terrain_ground.ground_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = 0
    terrain_ceiling.ceiling_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = int(3.0 / terrain_ceiling.vertical_scale)
    
    # Create square barrier around spawn area
    barrier_offset_px = int(0.5 / terrain_ground.horizontal_scale)  # 0.5m gap between spawn area and barriers
    
    # Calculate barrier positions around spawn area
    barrier_inner = spawn_half_size + barrier_offset_px
    barrier_outer = barrier_inner + barrier_width_px
    
    # North barrier
    if center_y + barrier_outer < terrain_ground.length:
        y_start = center_y + barrier_inner
        y_end = min(center_y + barrier_outer, terrain_ground.length)
        terrain_ground.ground_height_field_raw[:, y_start:y_end] = barrier_height_units
        terrain_ceiling.ceiling_height_field_raw[:, y_start:y_end] = barrier_height_units + gap_height_units
    
    # South barrier
    if center_y - barrier_outer >= 0:
        y_start = max(center_y - barrier_outer, 0)
        y_end = center_y - barrier_inner
        terrain_ground.ground_height_field_raw[:, y_start:y_end] = barrier_height_units
        terrain_ceiling.ceiling_height_field_raw[:, y_start:y_end] = barrier_height_units + gap_height_units
    
    # East barrier
    if center_x + barrier_outer < terrain_ground.width:
        x_start = center_x + barrier_inner
        x_end = min(center_x + barrier_outer, terrain_ground.width)
        terrain_ground.ground_height_field_raw[x_start:x_end, :] = barrier_height_units
        terrain_ceiling.ceiling_height_field_raw[x_start:x_end, :] = barrier_height_units + gap_height_units
    
    # West barrier
    if center_x - barrier_outer >= 0:
        x_start = max(center_x - barrier_outer, 0)
        x_end = center_x - barrier_inner
        terrain_ground.ground_height_field_raw[x_start:x_end, :] = barrier_height_units
        terrain_ceiling.ceiling_height_field_raw[x_start:x_end, :] = barrier_height_units + gap_height_units

    return terrain_ground, terrain_ceiling


def timber_piles_terrain(terrain_ground, terrain_ceiling, timber_spacing=1.0, timber_size=0.3, pile_height=1.2, hanging_obstacles=False, position_noise=0.2, height_noise=0.1):
    """
    Generate timber piles terrain using tiled layout with central platform
    
    Parameters:
        terrain_ground: ground SubTerrain object
        terrain_ceiling: ceiling SubTerrain object
        timber_spacing: spacing between timber centers [meters]
        timber_size: size of each timber pile [meters]
        pile_height: height of piles [meters]
        hanging_obstacles: whether to add hanging ceiling obstacles
        position_noise: random offset for tile positions [meters]
    """
    # Convert to discrete units
    timber_spacing_px = int(timber_spacing / terrain_ground.horizontal_scale)
    timber_size_px = int(timber_size / terrain_ground.horizontal_scale)
    pile_height_units = int(pile_height / terrain_ground.vertical_scale)
    position_noise_px = int(position_noise / terrain_ground.horizontal_scale)
    height_perturbation = int(height_noise / terrain_ground.vertical_scale)  # Small height perturbation for realism
    
    center_x = terrain_ground.width // 2
    center_y = terrain_ground.length // 2
    
    # Create large central spawn platform
    spawn_area_size_px = int(SPAWN_AREA_SIZE / terrain_ground.horizontal_scale)
    spawn_half_size = spawn_area_size_px // 2
    spawn_x1 = center_x - spawn_half_size
    spawn_x2 = center_x + spawn_half_size
    spawn_y1 = center_y - spawn_half_size
    spawn_y2 = center_y + spawn_half_size
    
    # Define exclusion zone around spawn area (no piles within this area)
    exclusion_radius_px = int((SPAWN_AREA_SIZE / 2 + 0.5) / terrain_ground.horizontal_scale)  # SPAWN_AREA_SIZE/2 + 0.5m buffer
    
    # Calculate grid dimensions based on terrain size and spacing
    grid_start_x = timber_size_px
    grid_end_x = terrain_ground.width - timber_size_px
    grid_start_y = timber_size_px  
    grid_end_y = terrain_ground.length - timber_size_px
    
    # Generate timber grid positions
    x_positions = np.arange(grid_start_x, grid_end_x, timber_spacing_px)
    y_positions = np.arange(grid_start_y, grid_end_y, timber_spacing_px)
    
    # Place timber piles in grid pattern
    for pile_x in x_positions:
        for pile_y in y_positions:
            # Add position noise if specified
            if position_noise_px > 0:
                noise_x = np.random.randint(-position_noise_px, position_noise_px + 1)
                noise_y = np.random.randint(-position_noise_px, position_noise_px + 1)
                pile_x_noisy = np.clip(pile_x + noise_x, timber_size_px, terrain_ground.width - timber_size_px - 1)
                pile_y_noisy = np.clip(pile_y + noise_y, timber_size_px, terrain_ground.length - timber_size_px - 1)
            else:
                pile_x_noisy = pile_x
                pile_y_noisy = pile_y
            
            # Check if pile is outside exclusion zone
            # dist_from_center = np.sqrt((pile_x_noisy - center_x)**2 + (pile_y_noisy - center_y)**2)
            # if dist_from_center < exclusion_radius_px:
            #     continue  # Too close to spawn area, skip this pile
            
            # Create square pile footprint
            half_size = timber_size_px // 2
            pile_x1 = max(0, pile_x_noisy - half_size)
            pile_x2 = min(terrain_ground.width, pile_x_noisy + half_size)
            pile_y1 = max(0, pile_y_noisy - half_size)
            pile_y2 = min(terrain_ground.length, pile_y_noisy + half_size)
            
            # Add height variation for realism
            pile_height_with_noise = pile_height_units + np.random.randint(-height_perturbation, height_perturbation + 1)
            terrain_ground.ground_height_field_raw[pile_x1:pile_x2, pile_y1:pile_y2] = pile_height_with_noise
            
            # Create corresponding ceiling depression (hanging obstacles)
            if hanging_obstacles:
                hanging_height = pile_height_units + int(0.3 / terrain_ceiling.vertical_scale)
                terrain_ceiling.ceiling_height_field_raw[pile_x1:pile_x2, pile_y1:pile_y2] = hanging_height

    # Create raised platform for spawn area
    platform_height = pile_height_units
    terrain_ground.ground_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = platform_height
    terrain_ceiling.ceiling_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = int(3.0 / terrain_ceiling.vertical_scale)

    return terrain_ground, terrain_ceiling


def confined_gap_terrain(terrain_ground, terrain_ceiling, gap_width=0.8, platform_size=1.0):
    """
    Generate confined gap terrain with central spawn platform and surrounding gaps
    
    Parameters:
        terrain_ground: ground SubTerrain object
        terrain_ceiling: ceiling SubTerrain object
        gap_width: width of the gaps around spawn area [meters]
        platform_size: size of central spawn platform [meters]
    """
    gap_width_px = int(gap_width / terrain_ground.horizontal_scale)
    
    center_x = terrain_ground.width // 2
    center_y = terrain_ground.length // 2
    
    # Create central spawn platform
    spawn_area_size_px = int(SPAWN_AREA_SIZE / terrain_ground.horizontal_scale)
    spawn_half_size = spawn_area_size_px // 2
    spawn_x1 = center_x - spawn_half_size
    spawn_x2 = center_x + spawn_half_size
    spawn_y1 = center_y - spawn_half_size
    spawn_y2 = center_y + spawn_half_size
    
    # Create raised central platform for spawn area
    platform_height = int(0.0 / terrain_ground.vertical_scale)
    terrain_ground.ground_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = platform_height
    terrain_ceiling.ceiling_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = int(2.0 / terrain_ceiling.vertical_scale)
    
    # Create gaps around the spawn platform
    gap_depth = int(1.0 / terrain_ground.vertical_scale)
    gap_offset_px = int(0.3 / terrain_ground.horizontal_scale)  # Small gap between platform and pits
    
    # Calculate gap positions around spawn area
    gap_inner = spawn_half_size + gap_offset_px
    gap_outer = gap_inner + gap_width_px
    
    # North gap
    if center_y + gap_outer < terrain_ground.length:
        y_start = center_y + gap_inner
        y_end = min(center_y + gap_outer, terrain_ground.length)
        terrain_ground.ground_height_field_raw[:, y_start:y_end] = -gap_depth
    
    # South gap
    if center_y - gap_outer >= 0:
        y_start = max(center_y - gap_outer, 0)
        y_end = center_y - gap_inner
        terrain_ground.ground_height_field_raw[:, y_start:y_end] = -gap_depth
    
    # East gap
    if center_x + gap_outer < terrain_ground.width:
        x_start = center_x + gap_inner
        x_end = min(center_x + gap_outer, terrain_ground.width)
        terrain_ground.ground_height_field_raw[x_start:x_end, :] = -gap_depth
    
    # West gap
    if center_x - gap_outer >= 0:
        x_start = max(center_x - gap_outer, 0)
        x_end = center_x - gap_inner
        terrain_ground.ground_height_field_raw[x_start:x_end, :] = -gap_depth
    
    # Create outer platforms beyond the gaps
    outer_platform_height = int(0.3 / terrain_ground.vertical_scale)
    outer_ceiling_height = int(1.8 / terrain_ceiling.vertical_scale)
    
    # Fill remaining areas with platforms
    mask = terrain_ground.ground_height_field_raw == 0  # Areas not yet modified
    terrain_ground.ground_height_field_raw[mask] = outer_platform_height
    terrain_ceiling.ceiling_height_field_raw[mask] = outer_ceiling_height

    return terrain_ground, terrain_ceiling


def column_obstacles_terrain(terrain_ground, terrain_ceiling, 
                             column_spacing=0.4, column_radius=0.1, 
                             column_height=0.8, hanging_length=0.8, density=0.7):
    """
    Generate column obstacles terrain with vertical columns and hanging obstacles for Franka robot testing
    
    Parameters:
        terrain_ground: ground SubTerrain object
        terrain_ceiling: ceiling SubTerrain object
        column_spacing: spacing between column centers [meters]
        column_radius: radius of cylindrical columns [meters]
        column_height: height of ground columns [meters]
        hanging_length: length of hanging obstacles from ceiling [meters]
        density: probability of placing an obstacle at each grid position [0-1]
    """
    spawn_area_size_franka = 0.3
    ceiling_height = 1.2  # Default ceiling height in meters
    pertub_unit = 10

    # Convert to discrete units
    column_spacing_px = int(column_spacing / terrain_ground.horizontal_scale)
    column_size_px = int(column_radius * 2 / terrain_ground.horizontal_scale)  # Use diameter as size
    column_height_units = int(column_height / terrain_ground.vertical_scale)
    hanging_length_units = int(hanging_length / terrain_ceiling.vertical_scale)
    
    center_x = terrain_ground.width // 2
    center_y = terrain_ground.length // 2
    
    # Create central spawn area (keep clear)
    spawn_area_size_px = int(spawn_area_size_franka / terrain_ground.horizontal_scale)
    spawn_half_size = spawn_area_size_px // 2
    
    # Keep spawn area flat (ground level 0)
    terrain_ground.ground_height_field_raw[:,:] = 0
    terrain_ceiling.ceiling_height_field_raw[:,:] = int(ceiling_height / terrain_ceiling.vertical_scale)
    
    # Define exclusion zone around spawn area
    exclusion_radius_px = int(spawn_area_size_franka / 2 / terrain_ground.horizontal_scale)
    
    # Calculate grid dimensions for column placement
    grid_start_x = column_size_px
    grid_end_x = terrain_ground.width - column_size_px
    grid_start_y = column_size_px
    grid_end_y = terrain_ground.length - column_size_px
    
    # Generate column grid positions
    x_positions = np.arange(grid_start_x, grid_end_x, column_spacing_px)
    y_positions = np.arange(grid_start_y, grid_end_y, column_spacing_px)


    # Place columns in grid pattern
    for col_x in x_positions:
        for col_y in y_positions:
            # Check if column is outside exclusion zone
            if np.abs(col_x - center_x) < exclusion_radius_px or np.abs(col_y - center_y) < exclusion_radius_px:
                continue  # Too close to spawn area, skip this column
            
            # Randomly place obstacles based on density
            if np.random.random() > density:
                continue
            
            # Create square column footprint (similar to timber piles)
            half_size = column_size_px // 2
            col_x1 = max(0, col_x - half_size)
            col_x2 = min(terrain_ground.width, col_x + half_size)
            col_y1 = max(0, col_y - half_size)
            col_y2 = min(terrain_ground.length, col_y + half_size)
            
            # Randomly choose between ground column, hanging obstacle, or both
            obstacle_type = np.random.choice(['ground', 'ceiling', 'both'], p=[0.3, 0.3, 0.4])

            if obstacle_type in ['ground', 'both']:
                # Create ground column with height variation
                column_height_with_noise = column_height_units + np.random.randint(-pertub_unit, pertub_unit + 1)
                terrain_ground.ground_height_field_raw[col_x1:col_x2, col_y1:col_y2] = column_height_with_noise
            
            if obstacle_type in ['ceiling', 'both']:
                # Create hanging obstacle from ceiling
                ceiling_base = int(ceiling_height / terrain_ceiling.vertical_scale)
                hanging_bottom = ceiling_base - hanging_length_units + np.random.randint(-pertub_unit, pertub_unit + 1)
                terrain_ceiling.ceiling_height_field_raw[col_x1:col_x2, col_y1:col_y2] = hanging_bottom
    
    return terrain_ground, terrain_ceiling


def wall_with_gap_terrain(terrain_ground, terrain_ceiling, 
                         gap_width=0.4, gap_height=0.5, gap_center_height=0.6,
                         wall_thickness=0.2):
    """
    Generate wall with gap terrain for Franka arm manipulation testing.
    The robot base is at the center, and the wall has a gap that the end-effector must navigate through.
    
    Parameters:
        terrain_ground: ground SubTerrain object
        terrain_ceiling: ceiling SubTerrain object
        gap_width: width of the gap opening [meters]
        gap_height: height of the gap opening [meters] 
        gap_center_height: height of gap center from ground [meters]
        wall_thickness: thickness of the wall [meters]
    """
    spawn_area_size_franka = 0.3  # Franka base area
    default_ceiling_height = 1.2  # Default ceiling height in meters
    default_ground_height = 0.0   # Default ground height
    
    # Convert to discrete units
    gap_width_px = int(gap_width / terrain_ground.horizontal_scale)
    wall_thickness_px = int(wall_thickness / terrain_ground.horizontal_scale)
    gap_center_height_units = int(gap_center_height / terrain_ground.vertical_scale)
    gap_height_units = int(gap_height / terrain_ceiling.vertical_scale)
    default_ground_units = int(default_ground_height / terrain_ground.vertical_scale)
    default_ceiling_units = int(default_ceiling_height / terrain_ceiling.vertical_scale)
    
    center_x = terrain_ground.width // 2
    center_y = terrain_ground.length // 2
    
    # Initialize entire terrain with default heights first
    terrain_ground.ground_height_field_raw[:, :] = default_ground_units
    terrain_ceiling.ceiling_height_field_raw[:, :] = default_ceiling_units
    
    # Create wall positioned at center_x (perpendicular to X-axis) - SWAPPED FROM Y
    wall_half_thickness = wall_thickness_px // 2
    wall_x1 = max(0, center_x - wall_half_thickness)
    wall_x2 = min(terrain_ground.width, center_x + wall_half_thickness)
    
    # Create the gap in the wall - SWAPPED X AND Y
    gap_half_width = gap_width_px // 2
    gap_y1 = max(0, center_y - gap_half_width)
    gap_y2 = min(terrain_ground.length, center_y + gap_half_width)
    
    # Calculate gap bounds in height
    gap_bottom_units = gap_center_height_units - gap_height_units // 2
    gap_top_units = gap_center_height_units + gap_height_units // 2
    
    # Create gap within the wall area - SWAPPED INDEXING ORDER
    # - Ground level at gap bottom
    # - Ceiling level at gap top
    terrain_ground.ground_height_field_raw[wall_x1:wall_x2, gap_y1:gap_y2] = gap_bottom_units
    terrain_ceiling.ceiling_height_field_raw[wall_x1:wall_x2, gap_y1:gap_y2] = gap_top_units
    
    # Set spawn area around robot base to default levels (ensure robot base area is accessible)
    spawn_area_size_px = int(spawn_area_size_franka / terrain_ground.horizontal_scale)
    spawn_half_size = spawn_area_size_px // 2
    spawn_x1 = center_x - spawn_half_size
    spawn_x2 = center_x + spawn_half_size
    spawn_y1 = center_y - spawn_half_size
    spawn_y2 = center_y + spawn_half_size
    
    # Ensure spawn area has default ground and ceiling heights
    terrain_ground.ground_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = default_ground_units
    # terrain_ceiling.ceiling_height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2] = default_ceiling_units
    
    return terrain_ground, terrain_ceiling


class SubTerrainConfined:
    """SubTerrain class for confined environments with ground and ceiling layers"""
    def __init__(self, terrain_name="confined_terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.ground_height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
        self.ceiling_height_field_raw = np.full((self.width, self.length), 
                                               int(3.0 / vertical_scale), dtype=np.int16)  # Default 3m ceiling


class TerrainConfined:
    """Terrain class for confined environments with ground and ceiling layers"""
    
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        
        if self.type in ["none", 'plane']:
            return
            
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        
        # Confined terrain proportions: [tunnel, barrier, timber_piles, confined_gap, column_obstacles, wall_with_gap]
        if hasattr(cfg, 'confined_terrain_proportions'):
            self.proportions = [np.sum(cfg.confined_terrain_proportions[:i+1]) 
                              for i in range(len(cfg.confined_terrain_proportions))]
        else:
            # Default proportions - added wall_with_gap
            self.proportions = [0.16, 0.32, 0.48, 0.64, 0.8, 1.0]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # Initialize both ground and ceiling height fields
        self.ground_height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.ceiling_height_field_raw = np.full((self.tot_rows, self.tot_cols), 
                                               int(3.0 / cfg.vertical_scale), dtype=np.int16)
        if hasattr(cfg, 'spawn_area_size'):
            global SPAWN_AREA_SIZE
            SPAWN_AREA_SIZE = cfg.spawn_area_size
            
        if cfg.curriculum:
            if hasattr(cfg, 'difficulty_scale'):
                self.curriculum(cfg.difficulty_scale)
            else:
                self.curriculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            if hasattr(cfg, 'difficulty_scale'):
                self.randomized_terrain(cfg.difficulty_scale)
            else:
                self.randomized_terrain()

        # Store height samples for compatibility
        self.heightsamples = self.ground_height_field_raw
        
        if self.type in ["trimesh", "confined_trimesh"]:
            self.vertices, self.triangles = convert_2layer_heightfield_to_trimesh(
                self.ground_height_field_raw,
                self.ceiling_height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold if hasattr(self.cfg, 'slope_treshold') else None
            )
        else:
            # Initialize empty vertices and triangles for compatibility
            self.vertices = None
            self.triangles = None

    def randomized_terrain(self, difficulty_scale=1.0):
        """Generate randomized confined terrain"""
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9]) * difficulty_scale
            terrain_ground, terrain_ceiling = self.make_confined_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain_ground, terrain_ceiling, i, j)

    def curriculum(self, difficulty_scale=1.0):
        """Generate curriculum-based confined terrain"""
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows * difficulty_scale
                choice = j / self.cfg.num_cols + 0.001
                terrain_ground, terrain_ceiling = self.make_confined_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain_ground, terrain_ceiling, i, j)

    def selected_terrain(self):
        """Generate selected confined terrain type"""
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain_ground = SubTerrainConfined("terrain_ground",
                                              width=self.width_per_env_pixels,
                                              length=self.length_per_env_pixels,
                                              vertical_scale=self.cfg.vertical_scale,
                                              horizontal_scale=self.cfg.horizontal_scale)
            
            terrain_ceiling = SubTerrainConfined("terrain_ceiling",
                                               width=self.width_per_env_pixels,
                                               length=self.length_per_env_pixels,
                                               vertical_scale=self.cfg.vertical_scale,
                                               horizontal_scale=self.cfg.horizontal_scale)

            # Apply selected terrain function
            terrain_ground, terrain_ceiling = eval(terrain_type)(terrain_ground, terrain_ceiling, 
                                                                **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(terrain_ground, terrain_ceiling, i, j)

    def make_confined_terrain(self, choice, difficulty):
        """Create confined terrain based on choice and difficulty"""
        terrain_ground = SubTerrainConfined("terrain_ground",
                                          width=self.width_per_env_pixels,
                                          length=self.length_per_env_pixels,
                                          vertical_scale=self.cfg.vertical_scale,
                                          horizontal_scale=self.cfg.horizontal_scale)
        
        terrain_ceiling = SubTerrainConfined("terrain_ceiling",
                                           width=self.width_per_env_pixels,
                                           length=self.length_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        
        # Scale parameters based on difficulty
        # Tunnel parameters
        tunnel_width = 1.5 * (1.2 - difficulty)
        tunnel_height = 0.8 * (1.1 - difficulty * 0.3)

        # Barrier parameters
        barrier_height = 0.2 + 0.1 * difficulty
        gap_height = 0.5 * (1 - difficulty)

        # Timber piles parameters
        pile_height = 0.1 + 0.0 * difficulty +0.5
        timber_spacing = 0.5 + 0.0 * difficulty + 0.0 # Smaller spacing = harder
        timber_size = 0.4 + 0.00 * difficulty
        position_noise = 0.0 + 0.0 * difficulty
        height_noise = 0.0 + 0.0 * difficulty

        # Gap parameters
        gap_width = 0.7 + 0.0 * difficulty
        platform_size = 1.0

        # Column obstacles parameters
        column_spacing = 0.3 - 0.0 * difficulty  # Smaller spacing = harder
        column_radius = 0.1 + 0.0 * difficulty  # Larger radius = harder
        column_height = 0.6 + 0.0 * difficulty   # Taller columns = harder
        hanging_length = 0.4 + 0.0 * difficulty  # Longer hanging obstacles = harder
        density = 0.8 + 0.0 * difficulty         # More obstacles = harder

        # Wall with gap parameters (fixed, not affected by difficulty)
        wall_gap_width = 2.0       # Width of gap opening [meters]
        wall_gap_height = 0.2       # Height of gap opening [meters] 
        wall_gap_center_height = 0.7  # Height of gap center from ground [meters]
        wall_thickness = 0.1   # Thickness of wall [meters]

        if choice < self.proportions[0]:
            # Tunnel terrain
            terrain_ground, terrain_ceiling = tunnel_terrain(
                terrain_ground, terrain_ceiling, 
                tunnel_width=tunnel_width, 
                tunnel_height=tunnel_height
            )
        elif choice < self.proportions[1]:
            # Barrier terrain  
            terrain_ground, terrain_ceiling = barrier_terrain(
                terrain_ground, terrain_ceiling,
                barrier_height=barrier_height,
                gap_height=gap_height
            )
        elif choice < self.proportions[2]:
            # Timber piles terrain
            terrain_ground, terrain_ceiling = timber_piles_terrain(
                terrain_ground, terrain_ceiling,
                timber_spacing=timber_spacing,
                timber_size=timber_size,
                pile_height=pile_height,
                position_noise=position_noise,
                height_noise=height_noise,
            )
        elif choice < self.proportions[3]:
            # Confined gap terrain
            terrain_ground, terrain_ceiling = confined_gap_terrain(
                terrain_ground, terrain_ceiling,
                gap_width=gap_width,
                platform_size=platform_size
            )
        elif choice < self.proportions[4]:
            # Column obstacles terrain
            terrain_ground, terrain_ceiling = column_obstacles_terrain(
                terrain_ground, terrain_ceiling,
                column_spacing=column_spacing,
                column_radius=column_radius,
                column_height=column_height,
                hanging_length=hanging_length,
                density=density
            )
        else:
            # Wall with gap terrain
            terrain_ground, terrain_ceiling = wall_with_gap_terrain(
                terrain_ground, terrain_ceiling,
                gap_width=wall_gap_width,
                gap_height=wall_gap_height,
                gap_center_height=wall_gap_center_height,
                wall_thickness=wall_thickness)

        return terrain_ground, terrain_ceiling

    def add_terrain_to_map(self, terrain_ground, terrain_ceiling, row, col):
        """Add terrain patches to the main height field maps"""
        i = row
        j = col
        # Map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels

        # Add ground terrain
        self.ground_height_field_raw[start_x:end_x, start_y:end_y] = terrain_ground.ground_height_field_raw
        
        # Add ceiling terrain
        self.ceiling_height_field_raw[start_x:end_x, start_y:end_y] = terrain_ceiling.ceiling_height_field_raw

        # Calculate environment origin (use ground level for origin)
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        check_rad = 0.05
        x1 = int((self.env_length/2. - check_rad) / terrain_ground.horizontal_scale)
        x2 = int((self.env_length/2. + check_rad) / terrain_ground.horizontal_scale)
        y1 = int((self.env_width/2. - check_rad) / terrain_ground.horizontal_scale)
        y2 = int((self.env_width/2. + check_rad) / terrain_ground.horizontal_scale)
        
        # Ensure indices are within bounds
        x1 = max(0, x1)
        x2 = min(terrain_ground.ground_height_field_raw.shape[0], x2)
        y1 = max(0, y1)
        y2 = min(terrain_ground.ground_height_field_raw.shape[1], y2)
        
        env_origin_z = np.max(terrain_ground.ground_height_field_raw[x1:x2, y1:y2]) * terrain_ground.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]