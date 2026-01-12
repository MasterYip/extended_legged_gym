#!/usr/bin/env python3
"""
Test script for SPHERICAL2 pattern in ray caster.
This script demonstrates the new SPHERICAL2 pattern with a ray caster in a 3D scene.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# Add parent directory to path to import from legged_gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import isaacgym before torch to avoid ImportError
import isaacgym
import torch

# Import LeggedRobot first to avoid circular import
from legged_gym.envs.base.legged_robot import LeggedRobot

# Now import ray_caster
from legged_gym.utils.ray_caster import (
    PatternType, RayCasterPatternCfg, RayCasterCfg, RayCaster
)


def create_test_mesh(save_path="/tmp/test_mesh.obj"):
    """Create a test mesh for ray casting.
    
    Args:
        save_path: Path to save the mesh to.
        
    Returns:
        Path to the saved mesh.
    """
    # Create a sample terrain with some features
    terrain = trimesh.creation.box(extents=[10, 10, 0.1])
    
    # Add some obstacles
    sphere1 = trimesh.creation.icosphere(radius=1.0)
    sphere1.apply_translation([3, 0, 1])
    
    sphere2 = trimesh.creation.icosphere(radius=0.8)
    sphere2.apply_translation([-2, 2, 0.8])
    
    cylinder = trimesh.creation.cylinder(radius=0.5, height=2)
    cylinder.apply_translation([0, -3, 1])
    
    # Combine meshes
    mesh = trimesh.util.concatenate([terrain, sphere1, sphere2, cylinder])
    
    # Save the mesh
    mesh.export(save_path)
    print(f"Saved test mesh to {save_path}")
    
    return save_path


def test_spherical2_ray_caster(mesh_path, num_points=64, polar_axis=None):
    """Test the SPHERICAL2 pattern with a ray caster.
    
    Args:
        mesh_path: Path to the mesh to ray cast against.
        num_points: Number of points in the SPHERICAL2 pattern.
        polar_axis: Custom polar axis direction [x, y, z].
    """
    # Create ray caster configuration
    polar_axis = polar_axis or [0.0, 0.0, 1.0]
    
    pattern_cfg = RayCasterPatternCfg(
        pattern_type=PatternType.SPHERICAL2,
        spherical2_num_points=num_points,
        spherical2_polar_axis=polar_axis
    )
    
    ray_caster_cfg = RayCasterCfg(
        pattern_cfg=pattern_cfg,
        mesh_paths=[mesh_path],
        max_distance=10.0,
        offset_pos=[0.0, 0.0, 0.0]
    )
    
    # Create ray caster
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ray_caster = RayCaster(ray_caster_cfg, num_envs=1, device=device)
    
    # Create sensor pose (position the sensor at a specific point)
    sensor_pos = torch.tensor([[0.0, 0.0, 1.5]], device=device)  # 1.5 units above the terrain
    
    # Test multiple orientations
    orientations = [
        [1.0, 0.0, 0.0, 0.0],  # Identity (look along x-axis)
        [0.7071, 0.0, 0.7071, 0.0],  # 90 degree rotation around y (look along z-axis)
        [0.7071, 0.0, 0.0, 0.7071],  # 90 degree rotation around z (look along y-axis)
    ]
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, quat in enumerate(orientations):
        # Set sensor orientation
        sensor_rot = torch.tensor([quat], device=device)
        
        # Update ray caster
        ray_caster.update(0.1, sensor_pos, sensor_rot)
        
        # Get ray caster data
        data = ray_caster.data
        
        # Convert to numpy for visualization
        ray_hits = data.ray_hits[0].cpu().numpy()
        ray_hits_found = data.ray_hits_found[0].cpu().numpy()
        sensor_pos_np = sensor_pos[0].cpu().numpy()
        
        # Create transformed ray directions for visualization
        ray_origins_local = ray_caster.ray_origins[0].cpu().numpy()
        ray_directions_local = ray_caster.ray_directions[0].cpu().numpy()
        
        # Create a 3D subplot
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Load the mesh for visualization
        mesh = trimesh.load(mesh_path)
        
        # Plot a subset of faces to make visualization clearer
        face_subset = np.random.choice(len(mesh.faces), size=min(500, len(mesh.faces)), replace=False)
        for face in mesh.faces[face_subset]:
            vertices = mesh.vertices[face]
            ax.plot3D(vertices[[0, 1, 2, 0], 0], 
                     vertices[[0, 1, 2, 0], 1], 
                     vertices[[0, 1, 2, 0], 2], 'gray', alpha=0.3)
        
        # Plot ray hits
        for j in range(len(ray_hits)):
            if ray_hits_found[j]:
                ax.plot([sensor_pos_np[0], ray_hits[j, 0]], 
                       [sensor_pos_np[1], ray_hits[j, 1]],
                       [sensor_pos_np[2], ray_hits[j, 2]], 'g-', alpha=0.5)
                ax.scatter(ray_hits[j, 0], ray_hits[j, 1], ray_hits[j, 2], c='r', s=20)
            else:
                # Calculate endpoint for rays that don't hit
                endpoint = sensor_pos_np + ray_directions_local[j] * ray_caster_cfg.max_distance
                ax.plot([sensor_pos_np[0], endpoint[0]], 
                       [sensor_pos_np[1], endpoint[1]],
                       [sensor_pos_np[2], endpoint[2]], 'b-', alpha=0.2)
        
        # Plot sensor position
        ax.scatter(sensor_pos_np[0], sensor_pos_np[1], sensor_pos_np[2], c='k', s=50)
        
        # Set axis limits
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 5])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        orientation_desc = {
            0: "Looking along X-axis",
            1: "Looking along Z-axis",
            2: "Looking along Y-axis"
        }
        
        ax.set_title(f"SPHERICAL2 Ray Casting\n{orientation_desc[i]}")
    
    plt.tight_layout()
    plt.show()


def main():
    # Create a test mesh
    mesh_path = create_test_mesh()
    
    # Test the SPHERICAL2 pattern with default polar axis
    print("Testing SPHERICAL2 pattern with default polar axis (Z-axis)")
    test_spherical2_ray_caster(mesh_path, num_points=64)
    
    # Test with X-axis as polar
    print("Testing SPHERICAL2 pattern with X-axis as polar")
    test_spherical2_ray_caster(mesh_path, num_points=64, polar_axis=[1.0, 0.0, 0.0])
    
    # Test with custom polar axis
    print("Testing SPHERICAL2 pattern with custom polar axis [1,1,1]")
    test_spherical2_ray_caster(mesh_path, num_points=64, polar_axis=[1.0, 1.0, 1.0])
    
    # Test with higher density
    print("Testing SPHERICAL2 pattern with higher density (128 points)")
    test_spherical2_ray_caster(mesh_path, num_points=128)


if __name__ == "__main__":
    main() 