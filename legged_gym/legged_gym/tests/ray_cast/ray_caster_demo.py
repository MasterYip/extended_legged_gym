#!/usr/bin/env python3

"""
Ray Caster Demo

This script demonstrates the usage of the RayCaster class without requiring 
an actual Isaac Gym environment. It creates a simple mesh and shows how to 
set up and use the ray caster to perform ray casting against it.
"""

import os
import isaacgym
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from legged_gym.envs import *
from legged_gym.utils.ray_caster import (
    RayCaster, 
    RayCasterCfg, 
    RayCasterPatternCfg,
    PatternType
)

def create_test_mesh(output_dir="/tmp"):
    """Create a test mesh and save it to a file.
    
    Args:
        output_dir: Directory to save the mesh to.
        
    Returns:
        Path to the saved mesh file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple mesh (a box with some height variation)
    vertices = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [-1, -1, 2],
        [1, -1, 1],
        [1, 1, 2],
        [-1, 1, 1]
    ], dtype=np.float32)
    
    # Define the faces
    faces = np.array([
        # Bottom
        [0, 1, 2],
        [0, 2, 3],
        # Sides
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
        # Top
        [4, 7, 6],
        [4, 6, 5]
    ], dtype=np.int32)
    
    # Create a trimesh mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Save the mesh to a file
    mesh_path = os.path.join(output_dir, "test_mesh.obj")
    mesh.export(mesh_path)
    
    return mesh_path, mesh

def demo_grid_pattern(mesh_path, device):
    """Demonstrate the ray caster with a grid pattern.
    
    Args:
        mesh_path: Path to the mesh file.
        device: Device to run on.
    """
    print("\nRunning grid pattern demo...")
    
    # Create ray caster configuration
    pattern_cfg = RayCasterPatternCfg(
        pattern_type=PatternType.GRID,
        grid_dims=(8, 8),
        grid_width=3.0,
        grid_height=3.0
    )
    
    ray_caster_cfg = RayCasterCfg(
        pattern_cfg=pattern_cfg,
        mesh_paths=[mesh_path],
        max_distance=10.0,
        offset_pos=[0.0, 0.0, 0.0]
    )
    
    # Create ray caster
    ray_caster = RayCaster(ray_caster_cfg, num_envs=1, device=device)
    
    # Create sensor pose (above the mesh, looking down)
    sensor_pos = torch.tensor([[0.0, 0.0, 5.0]], device=device)
    sensor_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)  # Looking straight down
    
    # Update ray caster
    ray_caster.update(0.1, sensor_pos, sensor_rot)
    
    return ray_caster

def demo_cone_pattern(mesh_path, device):
    """Demonstrate the ray caster with a cone pattern.
    
    Args:
        mesh_path: Path to the mesh file.
        device: Device to run on.
    """
    print("\nRunning cone pattern demo...")
    
    # Create ray caster configuration
    pattern_cfg = RayCasterPatternCfg(
        pattern_type=PatternType.CONE,
        cone_num_rays=24,
        cone_angle=30.0
    )
    
    ray_caster_cfg = RayCasterCfg(
        pattern_cfg=pattern_cfg,
        mesh_paths=[mesh_path],
        max_distance=10.0,
        offset_pos=[0.0, 0.0, 0.0]
    )
    
    # Create ray caster
    ray_caster = RayCaster(ray_caster_cfg, num_envs=1, device=device)
    
    # Create sensor pose (beside the mesh, looking towards it)
    sensor_pos = torch.tensor([[-3.0, 0.0, 1.0]], device=device)
    
    # Create rotation quaternion (pointing towards the mesh)
    # This is a 90-degree rotation around the y-axis
    sensor_rot = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]], device=device)
    
    # Update ray caster
    ray_caster.update(0.1, sensor_pos, sensor_rot)
    
    return ray_caster

def demo_spherical_pattern(mesh_path, device):
    """Demonstrate the ray caster with a spherical pattern.
    
    Args:
        mesh_path: Path to the mesh file.
        device: Device to run on.
    """
    print("\nRunning spherical pattern demo...")
    
    # Create ray caster configuration
    pattern_cfg = RayCasterPatternCfg(
        pattern_type=PatternType.SPHERICAL,
        spherical_num_azimuth=12,
        spherical_num_elevation=6
    )
    
    ray_caster_cfg = RayCasterCfg(
        pattern_cfg=pattern_cfg,
        mesh_paths=[mesh_path],
        max_distance=10.0,
        offset_pos=[0.0, 0.0, 0.0]
    )
    
    # Create ray caster
    ray_caster = RayCaster(ray_caster_cfg, num_envs=1, device=device)
    
    # Create sensor pose (inside the mesh)
    sensor_pos = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    sensor_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # Identity rotation
    
    # Update ray caster
    ray_caster.update(0.1, sensor_pos, sensor_rot)
    
    return ray_caster

def visualize_ray_caster(ray_caster, mesh, title):
    """Visualize the ray caster results.
    
    Args:
        ray_caster: Ray caster object.
        mesh: Trimesh mesh.
        title: Title for the plot.
    """
    # Get ray caster data
    data = ray_caster.data
    
    # Convert to numpy for visualization
    ray_hits = data.ray_hits[0].cpu().numpy()
    ray_hits_found = data.ray_hits_found[0].cpu().numpy()
    sensor_pos = data.pos[0].cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    ax.plot_trisurf(
        mesh.vertices[:, 0], 
        mesh.vertices[:, 1], 
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.2
    )
    
    # Plot the sensor position
    ax.scatter(
        sensor_pos[0],
        sensor_pos[1],
        sensor_pos[2],
        c='b',
        s=50,
        label='Sensor'
    )
    
    # Plot ray hits
    hits_valid = ray_hits[ray_hits_found]
    if len(hits_valid) > 0:
        ax.scatter(
            hits_valid[:, 0],
            hits_valid[:, 1],
            hits_valid[:, 2],
            c='r',
            s=20,
            label='Ray Hits'
        )
    
    # Draw rays
    for i in range(len(ray_hits)):
        if ray_hits_found[i]:
            end_point = ray_hits[i]
            ax.plot(
                [sensor_pos[0], end_point[0]],
                [sensor_pos[1], end_point[1]],
                [sensor_pos[2], end_point[2]],
                'g-',
                alpha=0.5
            )
        else:
            # For missed rays, we'll show them in a different color
            end_point = ray_hits[i]  # This is the ray at max distance
            ax.plot(
                [sensor_pos[0], end_point[0]],
                [sensor_pos[1], end_point[1]],
                [sensor_pos[2], end_point[2]],
                'r--',
                alpha=0.2
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    plt.tight_layout()
    
    return fig

def main():
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create test mesh
    mesh_path, mesh = create_test_mesh()
    print(f"Created test mesh at: {mesh_path}")
    
    # Run grid pattern demo
    ray_caster_grid = demo_grid_pattern(mesh_path, device)
    fig_grid = visualize_ray_caster(ray_caster_grid, mesh, "Ray Caster Demo - Grid Pattern")
    
    # Run cone pattern demo
    ray_caster_cone = demo_cone_pattern(mesh_path, device)
    fig_cone = visualize_ray_caster(ray_caster_cone, mesh, "Ray Caster Demo - Cone Pattern")
    
    # Run spherical pattern demo
    ray_caster_spherical = demo_spherical_pattern(mesh_path, device)
    fig_spherical = visualize_ray_caster(ray_caster_spherical, mesh, "Ray Caster Demo - Spherical Pattern")
    
    plt.show()

if __name__ == "__main__":
    main() 