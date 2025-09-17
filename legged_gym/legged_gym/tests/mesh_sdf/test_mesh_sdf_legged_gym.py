#!/usr/bin/env python3

"""
Mesh SDF Isaac Gym Demo

This script demonstrates mesh-based signed distance field (SDF) calculation in a running Isaac Gym simulation.
It loads a terrain mesh, performs SDF calculations, and visualizes the SDF values and gradients.
"""

# autopep8:off
import os
import sys
import numpy as np
import trimesh
from isaacgym import gymapi, gymutil
import torch
import math
import time

from legged_gym.envs import *
# Add parent directory to path to import mesh_sdf
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from legged_gym.utils.terrain_obj import TerrainObj
from legged_gym.utils.mesh_sdf import MeshSDF, MeshSDFCfg
from legged_gym.utils.gym_visualizer import GymVisualizer
# autopep8:on


def setup_gym():
    """Set up Isaac Gym simulation environment."""
    # Create gym instance
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(
        description="Mesh SDF Demo",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_points", "type": int, "default": 100, "help": "Number of query points to sample"},
            {"name": "--grid_mode", "action": "store_true", "help": "Use grid sampling instead of random points"},
            {"name": "--terrain_file", "type": str, "default": "", "help": "Custom terrain mesh file path"},
            {"name": "--cpu", "action": "store_true", "help": "Force CPU execution"}
        ]
    )

    # Set GPU by default, allow override with --cpu flag
    if not args.cpu:
        args.use_gpu = True
        args.use_gpu_pipeline = True

    # Configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # Set physics parameters
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # GPU sim options
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # Create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        return None, None, None, None, None

    # Create viewer
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 75.0
    camera_props.width = 1920
    camera_props.height = 1080
    viewer = gym.create_viewer(sim, camera_props)
    if viewer is None:
        print("*** Failed to create viewer")
        return None, None, None, None, None

    # Configure viewer camera
    cam_pos = gymapi.Vec3(20, 20, 30)
    cam_target = gymapi.Vec3(6, 6, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Set light parameters
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0, 0, -1))

    return gym, sim, viewer, args, sim_params


def create_ground_plane(gym, sim):
    """Create a ground plane in the simulation."""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
    plane_params.distance = 0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0

    gym.add_ground(sim, plane_params)


def create_test_mesh():
    """Create a test mesh and save it to a temporary file."""
    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)

    # Create a hilly terrain
    grid_size = 50
    terrain_size = 10.0  # terrain size in meters
    x = np.linspace(-terrain_size/2, terrain_size/2, grid_size)
    y = np.linspace(-terrain_size/2, terrain_size/2, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Create a terrain with multiple hills
    zz = np.zeros_like(xx)
    for i in range(len(x)):
        for j in range(len(y)):
            # Add multiple hills
            x_pos, y_pos = xx[i, j], yy[i, j]

            # Central hill
            dist1 = np.sqrt(x_pos**2 + y_pos**2)
            h1 = 2.0 * np.exp(-dist1**2 / 8.0)  # Gaussian hill

            # Smaller hills
            dist2 = np.sqrt((x_pos-3.0)**2 + (y_pos+2.0)**2)
            h2 = 1.0 * np.exp(-dist2**2 / 2.0)

            dist3 = np.sqrt((x_pos+3.0)**2 + (y_pos-2.0)**2)
            h3 = 1.5 * np.exp(-dist3**2 / 3.0)

            zz[i, j] = h1 + h2 + h3

    # Create mesh vertices and faces
    vertices = []
    for i in range(len(x)):
        for j in range(len(y)):
            vertices.append([xx[i, j], yy[i, j], zz[i, j]])

    vertices = np.array(vertices, dtype=np.float32)

    # Create faces (triangles)
    faces = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            idx = i * len(y) + j
            # Two triangles per grid cell
            faces.append([idx, idx + len(y), idx + len(y) + 1])
            faces.append([idx, idx + len(y) + 1, idx + 1])

    faces = np.array(faces, dtype=np.int32)

    # Create a trimesh mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save the mesh to a file
    mesh_path = os.path.join(output_dir, "sdf_demo_terrain.obj")
    mesh.export(mesh_path)

    print(f"Created test mesh at: {mesh_path}")
    return mesh_path, mesh


def sample_points(mode, num_points, terrain_bounds, device):
    """Sample points for SDF query."""
    x_min, y_min, z_min = terrain_bounds[0]
    x_max, y_max, z_max = terrain_bounds[1]
    half_x = (x_max - x_min) / 2.0
    half_y = (y_max - y_min) / 2.0
    x_min = x_min - half_x
    y_min = y_min - half_y
    x_max = x_max - half_x
    y_max = y_max - half_y
    
    # Calculate terrain center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    center_z = (z_min + z_max) / 2.0

    # Calculate terrain dimensions
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Add some margin around the terrain
    margin = 0.5
    x_min = center_x - (x_range/2 + margin)
    y_min = center_y - (y_range/2 + margin)
    z_min = center_z - (z_range/2 + margin)

    x_max = center_x + (x_range/2 + margin)
    y_max = center_y + (y_range/2 + margin)
    z_max = center_z + (z_range/2 + 3.0)  # Extra height margin

    print(f"Sampling points in region: [{x_min:.2f}, {y_min:.2f}, {z_min:.2f}] to [{x_max:.2f}, {y_max:.2f}, {z_max:.2f}]")
    print(f"Terrain center: [{center_x:.2f}, {center_y:.2f}, {center_z:.2f}]")

    if mode == "grid":
        # Create a 2D grid of points with varying heights
        grid_dim = int(np.sqrt(num_points))
        x = torch.linspace(x_min, x_max, grid_dim, device=device)
        y = torch.linspace(y_min, y_max, grid_dim, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # Create multiple horizontal slices at different heights
        num_heights = max(1, min(10, num_points // (grid_dim * grid_dim)))
        z_heights = torch.linspace(z_min, z_max, num_heights, device=device)

        points = []
        for z in z_heights:
            # Add a horizontal slice of points
            slice_points = torch.stack([grid_x.flatten(), grid_y.flatten(),
                                       torch.full_like(grid_x.flatten(), z)], dim=1)
            points.append(slice_points)

        points = torch.cat(points, dim=0)

        # If we have more points than requested, sample randomly
        if points.shape[0] > num_points:
            indices = torch.randperm(points.shape[0], device=device)[:num_points]
            points = points[indices]
    else:
        # Random sampling centered around terrain center
        points = torch.rand(num_points, 3, device=device)
        # Scale points to terrain bounds
        points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
        points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
        points[:, 2] = points[:, 2] * (z_max - z_min) + z_min

    return points


def visualize_sdf(visualizer, envs, points, sdf_values, sdf_gradients, nearest_points, env_idx=0):
    """Visualize SDF values, gradients, and nearest points."""
    # Convert data to CPU for visualization
    points_np = points.cpu().numpy()
    sdf_values_np = sdf_values.cpu().numpy()
    sdf_gradients_np = sdf_gradients.cpu().numpy()
    nearest_points_np = nearest_points.cpu().numpy()

    # Normalize SDF values for coloring
    sdf_min, sdf_max = np.min(sdf_values_np), np.max(sdf_values_np)
    normalized_sdf = (sdf_values_np - sdf_min) / (sdf_max - sdf_min + 1e-6)

    # Draw query points
    for i in range(len(points_np)):
        # Color based on SDF value (blue for negative/inside, red for positive/outside)
        if sdf_values_np[i] < 0:
            # Inside the mesh (blue to cyan)
            intensity = 0.5 + 0.5 * normalized_sdf[i]
            color = (0.0, intensity, 1.0)
        else:
            # Outside the mesh (yellow to red)
            intensity = 0.5 + 0.5 * (1.0 - normalized_sdf[i])
            color = (1.0, intensity, 0.0)

        # Draw the query point
        visualizer.draw_point(env_idx, points_np[i], color=color, size=0.05)

        # Draw SDF gradient arrows (showing the direction to the nearest surface)
        # Scale the arrows based on SDF value and gradient magnitude
        gradient_length = 0.3  # Fixed length for visibility
        gradient_magnitude = np.linalg.norm(sdf_gradients_np[i])

        if gradient_magnitude > 1e-6:  # Only draw gradients with meaningful direction
            # Draw the gradient direction
            gradient_normalized = sdf_gradients_np[i] / gradient_magnitude
            arrow_end = points_np[i] + gradient_normalized * gradient_length

            visualizer.draw_arrow(env_idx, points_np[i], arrow_end, width=0.01, color=color)

        # Draw lines to the nearest points on the surface
        visualizer.draw_line(env_idx, [points_np[i], nearest_points_np[i]], color=(0.5, 0.5, 0.5))

        # Draw the nearest points on the surface
        visualizer.draw_point(env_idx, nearest_points_np[i], color=(0.0, 1.0, 0.0), size=0.03)


def main():
    """Main function for the mesh SDF demo."""
    print("Starting Mesh SDF Isaac Gym Demo...")

    # Setup Isaac Gym
    gym, sim, viewer, args, sim_params = setup_gym()
    if gym is None:
        return

    # Create the terrain using the proper legged_gym approach
    terrain_cfg = gymutil.parse_arguments(
        description="Terrain Configuration",
        custom_parameters=[
            {"name": "--terrain_file", "type": str, "default": "", "help": "Custom terrain mesh file path"},
            {"name": "--terrain_type", "type": str, "default": "trimesh", "help": "Terrain type: plane, heightfield, trimesh"},
        ]
    )

    # Set up terrain configuration
    from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
    terrain_cfg = LeggedRobotCfg.terrain
    terrain_cfg.mesh_type = args.terrain_type if hasattr(args, 'terrain_type') else "trimesh"
    terrain_cfg.curriculum = False
    terrain_cfg.num_rows = 1
    terrain_cfg.num_cols = 1

    # Set up terrain file if provided
    if hasattr(args, 'terrain_file') and args.terrain_file and os.path.exists(args.terrain_file):
        terrain_cfg.terrain_file = args.terrain_file
        print(f"Using terrain file: {terrain_cfg.terrain_file}")
    else:
        # Create a test mesh and use it as terrain
        mesh_path, mesh = create_test_mesh()
        terrain_cfg.terrain_file = mesh_path
        print(f"Created test terrain at: {mesh_path}")

    # Before creating the TerrainObj, load the mesh and fix its normals
    mesh = trimesh.load(terrain_cfg.terrain_file)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

    # Ensure normals are facing upward (important for Isaac Gym visualization)
    print("Fixing mesh normals to ensure they face upward")
    # Calculate face normals
    face_normals = mesh.face_normals
    # Check if normals are predominantly pointing downward
    avg_normal_z = np.mean(face_normals[:, 2])
    print(f"Average normal Z direction before fixing: {avg_normal_z}")

    if avg_normal_z < 0:
        print("Normals are predominantly pointing downward, flipping faces")
        # Flip the faces by reversing vertex order
        mesh.faces = np.flip(mesh.faces, axis=1)
        # Recalculate normals
        mesh.fix_normals()
        avg_normal_z = np.mean(mesh.face_normals[:, 2])
        print(f"Average normal Z direction after fixing: {avg_normal_z}")

    # Save the fixed mesh
    fixed_mesh_path = "/tmp/fixed_" + os.path.basename(terrain_cfg.terrain_file)
    mesh.export(fixed_mesh_path)
    print(f"Saved fixed mesh to: {fixed_mesh_path}")

    # Update the terrain config to use the fixed mesh
    terrain_cfg.terrain_file = fixed_mesh_path

    # Always use terrain_obj for mesh SDF demo
    terrain_cfg.use_terrain_obj = True

    # Create the terrain object
    terrain = TerrainObj(terrain_cfg)
    print(f"Created TerrainObj with mesh from {terrain_cfg.terrain_file}")

    # Create the trimesh for Isaac Gym visualization
    # This follows _create_trimesh in legged_robot.py
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]
    tm_params.transform.p.x = -terrain.cfg.border_size
    tm_params.transform.p.y = -terrain.cfg.border_size

    gym.add_triangle_mesh(sim, terrain.vertices.flatten(), terrain.triangles.flatten(), tm_params)
    print("Added triangle mesh to simulation")

    # Create environments
    envs = []
    spacing = 4.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_envs_per_row = int(math.sqrt(args.num_envs))

    for i in range(args.num_envs):
        # Create environment
        env = gym.create_env(sim, env_lower, env_upper, num_envs_per_row)
        envs.append(env)

    print(f"Created {len(envs)} environments")

    # Initialize visualizer
    visualizer = GymVisualizer(gym, sim, viewer, envs)

    # Determine device based on args
    device = "cpu" if args.cpu else "cuda:0"
    print(f"Using device: {device}")

    # Setup mesh SDF using the fixed mesh
    mesh_sdf_cfg = MeshSDFCfg(
        mesh_paths=[terrain_cfg.terrain_file],
        max_distance=10.0,
        enable_caching=True
    )
    mesh_sdf = MeshSDF(mesh_sdf_cfg, device=device)

    # Get terrain bounds for query points
    terrain_bounds = terrain.terrain_mesh.bounds if hasattr(terrain, 'terrain_mesh') else mesh.bounds

    # Simulation loop
    frame = 0
    total_frames = 10000
    dt = sim_params.dt

    # Sampling mode
    sampling_mode = "grid" if args.grid_mode else "random"

    # Update interval (resample points every few seconds)
    update_interval = 5.0  # seconds
    last_update_time = -update_interval  # Force initial update
    current_points = None

    try:
        while not gym.query_viewer_has_closed(viewer) and frame < total_frames:
            # Step the simulation
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Update time
            t = frame * dt
            frame += 1

            # Resample points periodically
            if t - last_update_time >= update_interval:
                print(f"Sampling {args.num_points} points using {sampling_mode} mode...")
                # Sample query points
                current_points = sample_points(sampling_mode, args.num_points, terrain_bounds, device)

                # Perform SDF query
                start_time = time.time()
                sdf_values, sdf_gradients = mesh_sdf.query(current_points)
                nearest_points = mesh_sdf.nearest_points(current_points)
                query_time = time.time() - start_time

                print(f"SDF query completed in {query_time:.3f} seconds")
                print(f"SDF value range: [{sdf_values.min().item():.3f}, {sdf_values.max().item():.3f}]")

                last_update_time = t

            # Visualize query results
            if current_points is not None:
                # Clear previous visualization
                visualizer.clear()

                # Draw SDF query visualization
                visualize_sdf(visualizer, envs, current_points, sdf_values, sdf_gradients, nearest_points)

            # Update the viewer
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Demo completed.")


if __name__ == "__main__":
    main()
