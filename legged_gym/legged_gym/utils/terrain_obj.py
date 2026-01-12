'''
Author: Raymon Yip 2205929492@qq.com
Date: 2025-03-12 10:11:16
Description: file content
FilePath: /PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/legged_gym/utils/terrain_obj.py
LastEditTime: 2025-04-06 13:37:13
LastEditors: Raymon Yip
'''

import numpy as np
from numpy.random import choice
from scipy import interpolate
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
import trimesh
import torch
import os


class TerrainObj:
    """Terrain class for Obj Meshes

    public attributes used in LeggedRobot class:
        vertices: vertices of the terrain mesh
        triangles: triangles of the terrain mesh
        heightsamples: height samples of the terrain mesh
        cfg: terrain config
        tot_rows: total number of rows in the terrain
        tot_cols: total number of columns in the terrain
        env_origins: origins of the environments
        env_length: length of the environment
    """

    def verbose_print(self, msg):
        """self.verbose_prints the message if verbose is enabled."""
        if self.verbose:
            print(msg)

    def __init__(self, cfg: LeggedRobotCfg.terrain, verbose=False
                 ) -> None:
        self.cfg = cfg
        self.type = cfg.mesh_type
        self.verbose = verbose
        if self.type != "trimesh":
            raise ValueError("Only trimesh terrains are supported")

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # Load terrain mesh
        if os.path.isfile(self.cfg.terrain_file):
            self.terrain_mesh = trimesh.load(self.cfg.terrain_file)
            self.verbose_print(f"Loaded terrain mesh from {self.cfg.terrain_file}")
            self.verbose_print(f"Original mesh bounds: {self.terrain_mesh.bounds}")
        elif os.path.isfile(os.path.join(LEGGED_GYM_ROOT_DIR, self.cfg.terrain_file)):
            self.terrain_mesh = trimesh.load(os.path.join(LEGGED_GYM_ROOT_DIR, self.cfg.terrain_file))
            self.verbose_print(f"Loaded terrain mesh from {os.path.join(LEGGED_GYM_ROOT_DIR, self.cfg.terrain_file)}")
            self.verbose_print(f"Original mesh bounds: {self.terrain_mesh.bounds}")
        else:
            raise FileNotFoundError(f"Terrain mesh file not found: {self.cfg.terrain_file}")

        # Calculate original mesh dimensions and center
        original_bounds = self.terrain_mesh.bounds
        mesh_size_x = original_bounds[1][0] - original_bounds[0][0]
        mesh_size_y = original_bounds[1][1] - original_bounds[0][1]
        original_center = (original_bounds[0] + original_bounds[1]) / 2

        self.verbose_print(f"Original mesh size: x={mesh_size_x}, y={mesh_size_y}")
        self.verbose_print(f"Original mesh center: {original_center}")

        # IMPORTANT: In IsaacGym's _create_trimesh, the mesh origin is placed at:
        # tm_params.transform.p.x = -self.terrain.cfg.border_size
        # tm_params.transform.p.y = -self.terrain.cfg.border_size
        #
        # This means IsaacGym places the corner of the mesh at (-border_size, -border_size)
        # not at the center of the mesh. We need to account for this when centering.

        # Apply transformation to match IsaacGym's positioning
        # First, center the mesh at origin based on its bounds
        translation_to_center = np.eye(4)
        translation_to_center[0:2, 3] = -original_center[0:2]  # Center in XY, leave Z intact

        # Then, shift it to match IsaacGym's origin convention at (-border_size, -border_size)
        # Note: The trimesh is positioned in IsaacGym with its minimum corner at (-border_size, -border_size)
        # not with its center at origin. So we need to shift from center-at-origin to corner-at-offset.
        self.border_size = max(mesh_size_x, mesh_size_y) / 2
        self.cfg.border_size = self.border_size  # Save for IsaacGym to use

        # After centering, shift by (-border_size, -border_size) to match IsaacGym's positioning
        translation_to_corner = np.eye(4)
        translation_to_corner[0, 3] = self.border_size
        translation_to_corner[1, 3] = self.border_size

        # Apply both transformations (center first, then shift to corner)
        self.terrain_mesh.apply_transform(translation_to_center)
        self.terrain_mesh.apply_transform(translation_to_corner)

        # Verify the new mesh position
        new_bounds = self.terrain_mesh.bounds
        new_center = (new_bounds[0] + new_bounds[1]) / 2
        self.verbose_print(f"Transformed mesh bounds: {new_bounds}")
        self.verbose_print(f"Transformed mesh center: {new_center}")
        self.verbose_print(
            f"Minimum corner at: {new_bounds[0][0:2]}, should be close to: [{-self.border_size}, {-self.border_size}]")

        # Initialize ray mesh intersector for height evaluation
        self.ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.terrain_mesh)

        # Extract vertices and triangle indices from the transformed mesh
        self.vertices = np.array(self.terrain_mesh.vertices, dtype=np.float32)
        self.triangles = np.array(self.terrain_mesh.faces, dtype=np.uint32)

        # Calculate the environment origins based on IsaacGym's positioning
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # Calculate environment origins relative to the terrain mesh position
        # IsaacGym positions environments in a grid starting at the origin
        for i in range(cfg.num_rows):
            for j in range(cfg.num_cols):
                # Position environments in a grid with proper spacing
                env_origin_x = j * self.env_length - 0.5 * self.border_size
                env_origin_y = i * self.env_width - 0.5 * self.border_size

                # Get terrain height at this position (currently set to 0)
                height = 10.0

                # Store the environment origin
                self.env_origins[i, j] = [env_origin_x, env_origin_y, height]

        self.verbose_print(f"Environment origins shape: {self.env_origins.shape}")
        self.verbose_print(f"Environment origin corner: {self.env_origins[0, 0]}")
        self.verbose_print(f"Environment origin center: {self.env_origins[cfg.num_rows//2, cfg.num_cols//2]}")

        # Initialize heightsamples with zeros
        self.heightsamples = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

    def get_height(self, x, y, cast_dir=-1):
        """Get height at specific x,y coordinates on the terrain mesh.

        Args:
            x (float): X coordinate in world space
            y (float): Y coordinate in world space
            cast_dir (int): Direction to cast ray: -1 for downward, 1 for upward

        Returns:
            float: Height (z coordinate) at the specified position
        """
        # Use a default height of 0.0 if outside terrain or mesh isn't available
        if not hasattr(self, 'terrain_mesh'):
            return 0.0

        # Convert coordinates to mesh space
        # Account for the border offset in transformations
        # These coordinates need to be adjusted based on how the mesh was positioned
        mesh_x = x + self.border_size
        mesh_y = y + self.border_size

        # Check if the point is within mesh bounds
        bounds = self.terrain_mesh.bounds
        if (mesh_x < bounds[0][0] or mesh_x > bounds[1][0] or
                mesh_y < bounds[0][1] or mesh_y > bounds[1][1]):
            return 0.0

        # Create a ray for height sampling
        if cast_dir < 0:
            # Cast a ray from above the terrain straight down
            ray_origin = np.array([mesh_x, mesh_y, bounds[1][2] + 10.0])
        else:
            # Cast a ray from below the terrain straight up
            ray_origin = np.array([mesh_x, mesh_y, bounds[0][2] - 10.0])

        ray_direction = np.array([0.0, 0.0, cast_dir])

        # Intersect with the terrain mesh
        locations, _, _ = self.terrain_mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )

        # If there's an intersection, return the height
        if len(locations) > 0:
            return locations[0][2]

        # Fallback to a default height if no intersection
        return 0.0

    def get_heights_batch(self, positions, max_height=10.0, cast_dir=-1):
        """Get heights at multiple positions using ray casting.

        Args:
            positions (np.ndarray or torch.Tensor): Array of shape (N, 2) or (N, 3) containing x,y coordinates
                                                  If torch tensor, will be converted to numpy array
            max_height (float): Maximum height offset for ray origins
            cast_dir (int): Direction to cast ray: -1 for downward, 1 for upward

        Returns:
            np.ndarray: Array of shape (N,) containing heights at each position
        """
        # Convert torch tensor to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()

        # Ensure positions is at least 2D
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        # Extract x,y coordinates
        if positions.shape[1] >= 2:
            xy_positions = positions[:, :2]
        else:
            raise ValueError("positions must have at least 2 dimensions (x,y)")

        num_points = xy_positions.shape[0]

        # Adjust positions for ray casting by adding the border_size offset
        # This is necessary because the mesh is positioned with corner at (-border_size, -border_size)
        ray_xy_positions = xy_positions.copy()
        ray_xy_positions[:, 0] += self.border_size
        ray_xy_positions[:, 1] += self.border_size

        bounds = self.terrain_mesh.bounds

        # Set ray origins and directions based on cast direction
        if cast_dir < 0:
            # Create ray origins above the terrain for downward rays
            max_z_height = bounds[1][2] + max_height  # Add buffer above max terrain height
            ray_origins = np.column_stack([ray_xy_positions, np.full(num_points, max_z_height)])
        else:
            # Create ray origins below the terrain for upward rays
            min_z_height = bounds[0][2] - max_height  # Subtract buffer below min terrain height
            ray_origins = np.column_stack([ray_xy_positions, np.full(num_points, min_z_height)])

        # Create ray directions based on cast direction
        ray_directions = np.tile([0, 0, cast_dir], (num_points, 1))

        # Debug info
        if self.verbose:
            print(f"Ray origins bounds: {np.min(ray_origins[:, :2], axis=0)}, {np.max(ray_origins[:, :2], axis=0)}")
            print(f"Mesh bounds: {bounds}")
            print(f"Ray direction: [0, 0, {cast_dir}]")

        # Perform ray casting
        locations, _, _ = self.ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        # Initialize heights array with zeros
        heights = np.zeros(num_points)

        # For each point that had a hit, store the z-coordinate
        if len(locations) > 0:
            # Group hits by ray (some rays might have multiple hits)
            hits_per_ray = {}
            for hit in locations:
                # Find which ray this hit belongs to by matching x,y coordinates
                ray_idx = np.where((np.abs(ray_xy_positions[:, 0] - hit[0]) < 1e-6) &
                                   (np.abs(ray_xy_positions[:, 1] - hit[1]) < 1e-6))[0]
                if len(ray_idx) > 0:
                    ray_idx = ray_idx[0]
                    if ray_idx not in hits_per_ray:
                        hits_per_ray[ray_idx] = []
                    hits_per_ray[ray_idx].append(hit[2])

            # For each ray that had hits, use the appropriate hit point based on cast direction
            for ray_idx, hit_heights in hits_per_ray.items():
                if cast_dir < 0:
                    # For downward rays, use the highest hit point
                    heights[ray_idx] = max(hit_heights)
                else:
                    # For upward rays, use the lowest hit point
                    heights[ray_idx] = min(hit_heights)

        if self.verbose:
            print(f"Heights range: {np.min(heights) if len(heights) > 0 else 0} to "
                  f"{np.max(heights) if len(heights) > 0 else 0}")

        return heights
