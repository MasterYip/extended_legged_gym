#!/usr/bin/env python3
"""
Visualization script for ray casting patterns in the ray_caster module.
This script specifically visualizes the SPHERICAL2 pattern with different
densities and polar axis orientations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path to import from legged_gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import isaacgym before torch to avoid ImportError
import isaacgym
import torch

# Import LeggedRobot first to avoid circular import
from legged_gym.envs.base.legged_robot import LeggedRobot

# Now import ray_caster
from legged_gym.utils.ray_caster import PatternType, RayCasterPatternCfg


def visualize_spherical_pattern(pattern_type, num_points, polar_axis=None, title=None):
    """Visualize a spherical ray casting pattern.
    
    Args:
        pattern_type: Type of pattern (PatternType.SPHERICAL or PatternType.SPHERICAL2)
        num_points: Number of points for SPHERICAL2 or (num_azimuth, num_elevation) for SPHERICAL
        polar_axis: Custom polar axis direction [x, y, z]
        title: Custom title for the plot
    """
    # Set up pattern configuration
    if pattern_type == PatternType.SPHERICAL:
        num_azimuth, num_elevation = num_points
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.SPHERICAL,
            spherical_num_azimuth=num_azimuth,
            spherical_num_elevation=num_elevation
        )
    else:  # SPHERICAL2
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.SPHERICAL2,
            spherical2_num_points=num_points
        )
        if polar_axis is not None:
            pattern_cfg.spherical2_polar_axis = polar_axis

    # Generate the pattern
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ray_origins, ray_directions = pattern_cfg.create_pattern(device)

    # Convert to numpy for plotting
    ray_directions_np = ray_directions.cpu().numpy()

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ray directions as vectors from the origin
    for i in range(len(ray_directions_np)):
        direction = ray_directions_np[i]
        ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], 
                 color='b', alpha=0.6, arrow_length_ratio=0.1)

    # Plot a unit sphere (wireframe)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='r', alpha=0.1)

    # Plot polar axis if provided
    if polar_axis is not None:
        polar_axis = np.array(polar_axis) / np.linalg.norm(polar_axis)
        ax.quiver(0, 0, 0, polar_axis[0], polar_axis[1], polar_axis[2], 
                 color='r', linewidth=2, arrow_length_ratio=0.15)

    # Set axis limits and labels
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set plot title
    if title is None:
        if pattern_type == PatternType.SPHERICAL:
            title = f"SPHERICAL Pattern - {num_azimuth}x{num_elevation} rays"
        else:
            title = f"SPHERICAL2 Pattern - {num_points} rays"
            if polar_axis is not None:
                title += f" - Polar Axis: {polar_axis}"
    
    ax.set_title(title)
    
    return fig, ax


def compare_spherical_patterns():
    """Compare SPHERICAL and SPHERICAL2 patterns with different configurations."""
    plt.figure(figsize=(15, 10))
    
    # Standard SPHERICAL pattern (8x4 = 32 rays)
    plt.subplot(2, 2, 1, projection='3d')
    visualize_spherical_pattern(PatternType.SPHERICAL, (8, 4), title="SPHERICAL (8x4 = 32 rays)")
    
    # SPHERICAL2 pattern with 32 rays (default polar axis)
    plt.subplot(2, 2, 2, projection='3d')
    visualize_spherical_pattern(PatternType.SPHERICAL2, 32, title="SPHERICAL2 (32 rays, Z-axis)")
    
    # SPHERICAL2 pattern with custom polar axis
    plt.subplot(2, 2, 3, projection='3d')
    visualize_spherical_pattern(PatternType.SPHERICAL2, 32, polar_axis=[1.0, 0.0, 0.0], 
                               title="SPHERICAL2 (32 rays, X-axis)")
    
    # SPHERICAL2 pattern with higher density
    plt.subplot(2, 2, 4, projection='3d')
    visualize_spherical_pattern(PatternType.SPHERICAL2, 100, 
                               title="SPHERICAL2 (100 rays, Z-axis)")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Visualize a single SPHERICAL2 pattern
    fig, ax = visualize_spherical_pattern(
        PatternType.SPHERICAL2, 
        64,  # 64 points
        polar_axis=[0.0, 0.0, 1.0]  # Z-axis (default)
    )
    plt.show()
    
    # Visualize with X-axis as polar
    fig, ax = visualize_spherical_pattern(
        PatternType.SPHERICAL2, 
        64,  # 64 points
        polar_axis=[1.0, 0.0, 0.0]  # X-axis
    )
    plt.show()
    
    # Visualize with arbitrary polar axis
    fig, ax = visualize_spherical_pattern(
        PatternType.SPHERICAL2, 
        64,  # 64 points
        polar_axis=[1.0, 1.0, 1.0]  # Diagonal axis
    )
    plt.show()
    
    # Compare different pattern types and configurations
    compare_spherical_patterns() 