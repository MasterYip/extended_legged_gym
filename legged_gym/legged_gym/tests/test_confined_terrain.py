#!/usr/bin/env python3
"""
Test script for confined terrain system
This demonstrates how to use the new confined terrain with 2-layer heightfields
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain_confine import TerrainConfined
import numpy as np
import matplotlib.pyplot as plt


def test_confined_terrain():
    """Test the confined terrain generation and visualization"""
    
    # Configure terrain settings
    cfg = LeggedRobotCfg.terrain()
    cfg.mesh_type = 'confined_trimesh'
    cfg.horizontal_scale = 0.1
    cfg.vertical_scale = 0.005
    cfg.terrain_length = 8.0
    cfg.terrain_width = 8.0
    cfg.num_rows = 2
    cfg.num_cols = 2
    cfg.border_size = 2
    cfg.curriculum = False
    cfg.selected = False
    cfg.confined_terrain_proportions = [0.25, 0.5, 0.75, 1.0]
    
    # Create confined terrain
    num_robots = 4
    terrain = TerrainConfined(cfg, num_robots)
    
    print("Confined terrain generated successfully!")
    print(f"Ground heightfield shape: {terrain.ground_height_field_raw.shape}")
    print(f"Ceiling heightfield shape: {terrain.ceiling_height_field_raw.shape}")
    
    if terrain.vertices is not None:
        print(f"Number of vertices: {terrain.vertices.shape[0]}")
        print(f"Number of triangles: {terrain.triangles.shape[0]}")
    else:
        print("Trimesh conversion not performed (mesh_type not set to trimesh/confined_trimesh)")
    
    # Visualize the terrain
    if terrain.vertices is not None:
        fig = plt.figure(figsize=(15, 5))
        
        # Plot ground heightfield
        ax1 = fig.add_subplot(131)
        ground_heights = terrain.ground_height_field_raw * cfg.vertical_scale
        im1 = ax1.imshow(ground_heights, cmap='terrain', origin='lower')
        ax1.set_title('Ground Layer')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Height (m)')
        
        # Plot ceiling heightfield
        ax2 = fig.add_subplot(132)
        ceiling_heights = terrain.ceiling_height_field_raw * cfg.vertical_scale
        im2 = ax2.imshow(ceiling_heights, cmap='viridis', origin='lower')
        ax2.set_title('Ceiling Layer')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Height (m)')
        
        # Plot 3D view of vertices
        ax3 = fig.add_subplot(133, projection='3d')
        vertices = terrain.vertices
        
        # Sample vertices for visualization (to avoid overcrowding)
        sample_idx = np.random.choice(len(vertices), min(1000, len(vertices)), replace=False)
        sampled_vertices = vertices[sample_idx]
        
        # Separate ground and ceiling vertices
        mid_z = np.median(sampled_vertices[:, 2])
        ground_mask = sampled_vertices[:, 2] <= mid_z
        ceiling_mask = sampled_vertices[:, 2] > mid_z
        
        # Plot ground vertices in brown
        if np.any(ground_mask):
            ax3.scatter(sampled_vertices[ground_mask, 0], 
                       sampled_vertices[ground_mask, 1], 
                       sampled_vertices[ground_mask, 2], 
                       c='brown', s=1, alpha=0.6, label='Ground')
        
        # Plot ceiling vertices in blue
        if np.any(ceiling_mask):
            ax3.scatter(sampled_vertices[ceiling_mask, 0], 
                       sampled_vertices[ceiling_mask, 1], 
                       sampled_vertices[ceiling_mask, 2], 
                       c='blue', s=1, alpha=0.6, label='Ceiling')
        
        ax3.set_title('3D Mesh Preview')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Visualization complete!")
        print("\nTo use confined terrain in your robot environment:")
        print("1. Set cfg.terrain.mesh_type = 'confined_trimesh'")
        print("2. Adjust cfg.terrain.confined_terrain_proportions for different terrain types")
        print("3. The system will generate both ground and ceiling surfaces")
        print("4. Terrain types include: tunnel, barrier, timber_piles, confined_gap")
    
    return terrain


def test_individual_terrain_types():
    """Test individual confined terrain generation functions"""
    from legged_gym.utils.terrain_confine import (
        SubTerrainConfined, tunnel_terrain, barrier_terrain, 
        timber_piles_terrain, confined_gap_terrain
    )
    
    print("\nTesting individual terrain types...")
    
    # Create base terrain objects
    width = 100
    length = 100
    
    terrain_ground = SubTerrainConfined("test_ground", width, length)
    terrain_ceiling = SubTerrainConfined("test_ceiling", width, length)
    
    # Test tunnel terrain
    tunnel_terrain(terrain_ground, terrain_ceiling, tunnel_width=2.0, tunnel_height=2.5)
    print("✓ Tunnel terrain generated")
    
    # Reset for next test
    terrain_ground = SubTerrainConfined("test_ground", width, length)
    terrain_ceiling = SubTerrainConfined("test_ceiling", width, length)
    
    # Test barrier terrain
    barrier_terrain(terrain_ground, terrain_ceiling, barrier_height=1.5, gap_height=0.8)
    print("✓ Barrier terrain generated")
    
    # Reset for next test  
    terrain_ground = SubTerrainConfined("test_ground", width, length)
    terrain_ceiling = SubTerrainConfined("test_ceiling", width, length)
    
    # Test timber piles terrain
    timber_piles_terrain(terrain_ground, terrain_ceiling, timber_spacing=1.0, timber_size=0.3, position_noise=0.2)
    print("✓ Timber piles terrain generated")
    
    # Reset for next test
    terrain_ground = SubTerrainConfined("test_ground", width, length)
    terrain_ceiling = SubTerrainConfined("test_ceiling", width, length)
    
    # Test confined gap terrain
    confined_gap_terrain(terrain_ground, terrain_ceiling)
    print("✓ Confined gap terrain generated")
    
    print("All individual terrain types tested successfully!")


if __name__ == "__main__":
    print("Testing Confined Terrain System")
    print("=" * 40)
    
    # Test the full terrain system
    terrain = test_confined_terrain()
    
    # Test individual terrain functions
    test_individual_terrain_types()
    
    print("\n" + "=" * 40)
    print("Confined terrain system test completed successfully!")
    print("\nKey features implemented:")
    print("- 2-layer heightfield system (ground + ceiling)")
    print("- 4 terrain types: tunnel, barrier, timber_piles, confined_gap")
    print("- convert_2layer_heightfield_to_trimesh function")
    print("- Integration with existing legged_gym terrain system")
    print("- Configurable through LeggedRobotCfg.terrain")