# autopep8:off
import isaacgym  # Required for isaacgym.torch_utils
from isaacgym.torch_utils import quat_apply
from legged_gym.envs.base.base_task import BaseTask
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add the parent directory to path to import from legged_gym
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from legged_gym.utils.mesh_sdf import MeshSDF, MeshSDFCfg
# autopep8:on


def test_mesh_sdf_basic():
    """Basic test for MeshSDF class with a sphere."""
    print("Running basic MeshSDF test...")

    # Create a sample mesh (sphere)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_path = "/tmp/test_sphere.obj"
    sphere.export(mesh_path)

    # Create MeshSDF configuration
    mesh_sdf_cfg = MeshSDFCfg(
        mesh_paths=[mesh_path],
        max_distance=10.0,
        enable_caching=True
    )

    # Create MeshSDF
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mesh_sdf = MeshSDF(mesh_sdf_cfg, device=device)

    # Test a single point (should be close to -1.0 for a point at the origin)
    origin_point = torch.zeros(1, 3, device=device)
    sdf_value, sdf_gradient = mesh_sdf.query(origin_point)

    print(f"SDF value at origin: {sdf_value.item():.4f} (should be close to -1.0 for a unit sphere)")
    print(f"SDF gradient at origin: {sdf_gradient.squeeze().cpu().numpy()}")

    return mesh_sdf, mesh_path


def test_mesh_sdf_batch():
    """Test batch processing with MeshSDF."""
    print("\nRunning batch processing test...")

    # Create a sample mesh (sphere)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_path = "/tmp/test_sphere.obj"
    sphere.export(mesh_path)

    # Create MeshSDF configuration
    mesh_sdf_cfg = MeshSDFCfg(
        mesh_paths=[mesh_path],
        max_distance=10.0,
        enable_caching=False  # Disable caching to test performance
    )

    # Create MeshSDF
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mesh_sdf = MeshSDF(mesh_sdf_cfg, device=device)

    # Create a batch of points
    batch_size = 10
    num_points = 1000
    # Random points in a cube from -2 to 2
    points = torch.rand(batch_size, num_points, 3, device=device) * 4 - 2

    # Query SDF values and gradients
    import time
    start_time = time.time()
    sdf_values, sdf_gradients = mesh_sdf.query(points)
    end_time = time.time()

    print(f"Processed {batch_size * num_points} points in {end_time - start_time:.4f} seconds")
    print(f"Shape of SDF values: {sdf_values.shape}")
    print(f"Shape of SDF gradients: {sdf_gradients.shape}")

    # Verify some expected values
    inside_mask = sdf_values < 0
    outside_mask = sdf_values > 0
    print(f"Number of points inside the sphere: {inside_mask.sum().item()}")
    print(f"Number of points outside the sphere: {outside_mask.sum().item()}")

    return mesh_sdf, mesh_path


def visualize_mesh_sdf(mesh_sdf, mesh_path):
    """Visualize the MeshSDF results."""
    print("\nVisualizing MeshSDF results...")

    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # Create a grid of points to query
    device = next(iter(mesh_sdf.meshes.values())).points.device
    device = "cuda" if device == "cuda" else "cpu"
    torch_device = "cuda:0" if device == "cuda" else "cpu"

    # Create a 2D grid of points on the XZ plane
    resolution = 50
    x = torch.linspace(-2, 2, resolution, device=torch_device)
    z = torch.linspace(-2, 2, resolution, device=torch_device)
    grid_x, grid_z = torch.meshgrid(x, z, indexing='ij')
    y = torch.zeros_like(grid_x)

    points = torch.stack([grid_x.flatten(), y.flatten(), grid_z.flatten()], dim=1)

    # Query SDF values and gradients
    sdf_values, sdf_gradients = mesh_sdf.query(points)

    # Reshape for visualization
    sdf_values_grid = sdf_values.reshape(resolution, resolution).cpu().numpy()

    # Calculate nearest points on the surface
    nearest_points = mesh_sdf.nearest_points(points)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot SDF values as a contour map
    contour = ax1.contourf(
        grid_x.cpu().numpy(),
        grid_z.cpu().numpy(),
        sdf_values_grid,
        levels=50,
        cmap='viridis'
    )
    plt.colorbar(contour, ax=ax1)

    # Draw the zero level set
    zero_contour = ax1.contour(
        grid_x.cpu().numpy(),
        grid_z.cpu().numpy(),
        sdf_values_grid,
        levels=[0],
        colors='r',
        linewidths=3
    )

    ax1.set_title('SDF Values on XZ Plane (Y=0)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_aspect('equal')

    # Plot the mesh cross-section and nearest points in 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot original mesh
    ax2.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.2,
        color='blue'
    )

    # Sample some points to show nearest surface points
    sample_indices = torch.randperm(len(points))[:100]
    sample_points = points[sample_indices].cpu().numpy()
    sample_nearest = nearest_points[sample_indices].cpu().numpy()

    # Plot sampled points and their nearest surface points with connecting lines
    ax2.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        sample_points[:, 2],
        c='r',
        s=20,
        label='Query Points'
    )

    ax2.scatter(
        sample_nearest[:, 0],
        sample_nearest[:, 1],
        sample_nearest[:, 2],
        c='g',
        s=20,
        label='Nearest Surface Points'
    )

    # Draw lines connecting query points to their nearest points
    for i in range(len(sample_points)):
        ax2.plot(
            [sample_points[i, 0], sample_nearest[i, 0]],
            [sample_points[i, 1], sample_nearest[i, 1]],
            [sample_points[i, 2], sample_nearest[i, 2]],
            'k-',
            alpha=0.3
        )

    ax2.set_title('3D Visualization of Nearest Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('mesh_sdf_visualization.png')
    print("Visualization saved as 'mesh_sdf_visualization.png'")
    plt.show()


if __name__ == "__main__":
    # Run basic test
    mesh_sdf, mesh_path = test_mesh_sdf_basic()

    # Run batch processing test
    test_mesh_sdf_batch()

    # Visualize results
    visualize_mesh_sdf(mesh_sdf, mesh_path)
