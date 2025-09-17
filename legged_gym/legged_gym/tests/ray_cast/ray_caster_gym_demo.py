#!/usr/bin/env python3

"""
Ray Caster Isaac Gym Demo

This script demonstrates ray casting in a running Isaac Gym simulation.
It loads a terrain mesh, performs ray casting, and visualizes the hits.
"""

import os
import numpy as np
import trimesh
from isaacgym import gymapi, gymutil
import torch
import math

from legged_gym.envs import *
from legged_gym.utils.gym_visualizer import GymVisualizer
from legged_gym.utils.ray_caster import RayCaster, RayCasterCfg, RayCasterPatternCfg, PatternType


def setup_gym():
    """Set up Isaac Gym simulation environment."""
    # Create gym instance
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(
        description="Ray Caster Demo",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
            {"name": "--pattern", "type": str, "default": "cone",
             "help": "Ray pattern (single, grid, cone, spherical)"},
            {"name": "--num_rays", "type": int, "default": 24, "help": "Number of rays for cone/spherical pattern"},
            {"name": "--angle", "type": float, "default": 30.0, "help": "Cone angle in degrees"},
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


def create_terrain_asset(gym, sim, mesh_path):
    """Create a terrain asset from a mesh file."""
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.thickness = 0.001

    # Load the mesh
    terrain_asset = gym.load_asset(sim, os.path.dirname(mesh_path), os.path.basename(mesh_path), asset_options)

    return terrain_asset


def create_box_asset(gym, sim, width=1.0, height=0.5, depth=1.5):
    """Create a box asset for testing ray casting."""
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True

    # Create the asset
    box_asset = gym.create_box(sim, width, depth, height, asset_options)

    return box_asset


def create_environments(gym, sim, num_envs, terrain_asset=None, box_asset=None):
    """Create simulation environments."""
    # Set up the env grid
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Create environments
    envs = []
    terrain_handles = []
    box_handles = []
    num_envs_per_row = int(math.sqrt(num_envs))

    for i in range(num_envs):
        # Create environment
        env = gym.create_env(sim, env_lower, env_upper, num_envs_per_row)
        envs.append(env)

        # Add terrain if available
        if terrain_asset is not None:
            terrain_pose = gymapi.Transform()
            terrain_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            terrain_pose.r = gymapi.Quat(0, 0, 0, 1)

            terrain_handle = gym.create_actor(env, terrain_asset, terrain_pose, f"terrain_{i}", i, 1)
            gym.set_rigid_body_color(env, terrain_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.5, 0.3))
            terrain_handles.append(terrain_handle)

        # Add box if available
        if box_asset is not None:
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(0.0, 0.0, 0.25)  # Slightly above ground
            box_pose.r = gymapi.Quat(0, 0, 0, 1)

            box_handle = gym.create_actor(env, box_asset, box_pose, f"box_{i}", i, 1)
            gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.2))
            box_handles.append(box_handle)

    return envs, terrain_handles, box_handles


def create_test_mesh():
    """Create a test mesh and save it to a temporary file."""
    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)

    # Create a simple terrain with a hill
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    xx, yy = np.meshgrid(x, y)

    # Create a terrain with a central hill
    zz = np.zeros_like(xx)
    for i in range(len(x)):
        for j in range(len(y)):
            dist = np.sqrt(xx[i, j]**2 + yy[i, j]**2)
            zz[i, j] = 1.0 * np.exp(-dist**2 / 8.0)  # Gaussian hill

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
    mesh_path = os.path.join(output_dir, "terrain_demo.obj")
    mesh.export(mesh_path)

    return mesh_path, mesh


def setup_ray_caster(pattern_type, num_rays, angle, mesh_path, device, num_envs):
    """Set up ray caster with specified pattern."""
    # Create pattern configuration
    if pattern_type == "single":
        pattern_cfg = RayCasterPatternCfg(pattern_type=PatternType.SINGLE_RAY)
    elif pattern_type == "grid":
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.GRID,
            grid_dims=(5, 5),
            grid_width=2.0,
            grid_height=2.0
        )
    elif pattern_type == "cone":
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.CONE,
            cone_num_rays=num_rays,
            cone_angle=angle
        )
    elif pattern_type == "spherical":
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.SPHERICAL,
            spherical_num_azimuth=num_rays,
            spherical_num_elevation=15
        )
    elif pattern_type == "spherical2":
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.SPHERICAL2,
            spherical2_num_points=num_rays,
            spherical2_polar_axis=[0.0, 0.0, 1.0]
        )
    else:
        print(f"Unknown pattern type: {pattern_type}. Using cone pattern.")
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.CONE,
            cone_num_rays=num_rays,
            cone_angle=angle
        )

    # Create ray caster configuration
    ray_caster_cfg = RayCasterCfg(
        pattern_cfg=pattern_cfg,
        mesh_paths=[mesh_path],
        max_distance=5.0,
        offset_pos=[0.0, 0.0, 0.0],  # Offset from the reference point
        attach_yaw_only=True  # FIXME: False has problems
    )

    # Create ray caster
    ray_caster = RayCaster(ray_caster_cfg, num_envs, device)

    return ray_caster


def main():
    """Main function for the ray caster demo."""
    print("Starting Ray Caster Isaac Gym Demo...")

    # Setup Isaac Gym
    gym, sim, viewer, args, sim_params = setup_gym()
    if gym is None:
        return

    # Create ground plane
    create_ground_plane(gym, sim)

    # Create a box for visualization
    # box_asset = create_box_asset(gym, sim, width=4.0, height=0.5, depth=4.0)

    # Create environments
    envs, _, box_handles = create_environments(
        gym, sim, args.num_envs, terrain_asset=None, box_asset=None)

    # Create a mesh for raycast testing (this will only be used by WARP, not displayed in Isaac Gym)
    # Create a simple box mesh for raycasting
    box = trimesh.creation.box(extents=[2.0, 2.0, 5.0])
    box.vertices[:, 2] += 0.4  # Raise it to be slightly above ground

    # Add a hill to the box
    center_x, center_y = 0, 0
    hill_radius = 2.0
    hill_height = 1.0

    for i, vertex in enumerate(box.vertices):
        x, y = vertex[0], vertex[1]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist < hill_radius:
            # Add a Gaussian hill
            z_offset = hill_height * np.exp(-dist**2 / (2 * (hill_radius/3)**2))
            box.vertices[i, 2] += z_offset

    # Save mesh for raycasting
    mesh_path = "/tmp/raycast_terrain.obj"
    box.export(mesh_path)
    print(f"Created test mesh at: {mesh_path}")

    # Initialize visualizer
    visualizer = GymVisualizer(gym, sim, viewer, envs)

    # Determine device based on args
    device = "cpu" if args.cpu else "cuda:0"
    print(f"Using device: {device}")

    # Setup ray caster
    ray_caster = setup_ray_caster(
        args.pattern, args.num_rays, args.angle, mesh_path, device, args.num_envs)

    # Set up simulation loop
    dt = sim_params.dt

    # Create sensor pose arrays
    sensor_pos_np = np.zeros((args.num_envs, 3), dtype=np.float32)
    sensor_rot_np = np.zeros((args.num_envs, 4), dtype=np.float32)
    sensor_rot_np[:, 3] = 1.0  # Identity quaternion (w,x,y,z)

    # Create tensors on specified device
    sensor_pos = torch.tensor(sensor_pos_np, dtype=torch.float32, device=device)
    sensor_rot = torch.tensor(sensor_rot_np, dtype=torch.float32, device=device)

    # Simulation loop
    frame = 0
    total_frames = 10000
    radius = 3.0
    height = 1.5

    # Get wireframe points for visualization
    wireframe_points = box.vertices.copy()

    try:
        while not gym.query_viewer_has_closed(viewer) and frame < total_frames:
            # Step the simulation
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Update time
            t = frame * dt
            frame += 1

            # Update ray caster sensor pose (moving in a circle and rotating)
            angle = t * 0.5  # Rotate at 0.5 rad/s
            angle_phases = torch.linspace(0, 2 * math.pi, args.num_envs, device=device)

            with torch.no_grad():
                # Update positions directly on the device if using CUDA
                if device == "cuda:0":
                    # Batch process
                    sensor_pos[:, 0] = radius * torch.cos(angle_phases + angle)
                    sensor_pos[:, 1] = radius * torch.sin(angle_phases + angle)
                    sensor_pos[:, 2] = height
                    # Compute yaw to look at center
                    look_dir = -sensor_pos[:, :2]
                    look_dir_norm = torch.norm(look_dir, dim=1)
                    look_dir[look_dir_norm > 1e-6] /= look_dir_norm[look_dir_norm > 1e-6].unsqueeze(1)
                    yaw = torch.atan2(look_dir[:, 1], look_dir[:, 0])
                    sensor_rot[:, 3] = torch.cos(yaw / 2)  # w
                    sensor_rot[:, 2] = torch.sin(yaw / 2)  # z

                else:
                    # Use CPU arrays for calculations
                    sensor_pos_np.fill(0.0)
                    sensor_rot_np.fill(0.0)
                    sensor_rot_np[:, 3] = 1.0  # Reset to identity quaternion
                    for i in range(args.num_envs):
                        # Position (circular path)
                        sensor_pos_np[i, 0] = radius * math.cos(angle)
                        sensor_pos_np[i, 1] = radius * math.sin(angle)
                        sensor_pos_np[i, 2] = height

                        # Rotation (looking at center)
                        # Calculate rotation from [0,0,1] to look_dir
                        look_dir = np.array([-sensor_pos_np[i, 0], -sensor_pos_np[i, 1], 0.0])
                        if np.linalg.norm(look_dir) > 1e-6:  # Avoid division by zero
                            look_dir = look_dir / np.linalg.norm(look_dir)
                            yaw = math.atan2(look_dir[1], look_dir[0])
                            sensor_rot_np[i, 2] = math.sin(yaw/2)  # w
                            sensor_rot_np[i, 3] = math.cos(yaw/2)  # z

                    # Create new CPU tensors
                    sensor_pos = torch.tensor(sensor_pos_np, device=device)
                    sensor_rot = torch.tensor(sensor_rot_np, device=device)

            # Update ray caster
            ray_caster.update(dt, sensor_pos, sensor_rot)

            # Get ray caster data
            data = ray_caster.data

            # Clear previous visualization
            visualizer.clear()

            # Draw wireframe of the box mesh to show what we're raycasting against
            for env_idx in range(args.num_envs):
                # Draw the wireframe of the box
                for face in box.faces:
                    p1 = wireframe_points[face[0]]
                    p2 = wireframe_points[face[1]]
                    p3 = wireframe_points[face[2]]

                    # Draw the triangle edges
                    visualizer.draw_line(env_idx, [p1, p2], color=(0.2, 0.2, 1.0))
                    visualizer.draw_line(env_idx, [p2, p3], color=(0.2, 0.2, 1.0))
                    visualizer.draw_line(env_idx, [p3, p1], color=(0.2, 0.2, 1.0))

            # Visualize ray caster results
            for env_idx in range(args.num_envs):
                # Sensor position in world space
                sensor_position = data.pos[env_idx].cpu().numpy()
                # Draw sensor position
                visualizer.draw_point(env_idx, sensor_position, color=(0, 0, 1), size=0.1)

                # Draw a coordinate frame at the sensor position
                visualizer.draw_frame_from_quat(env_idx,
                                                [sensor_rot_np[env_idx, 0],  # x
                                                 sensor_rot_np[env_idx, 1],  # y
                                                 sensor_rot_np[env_idx, 2],  # z
                                                 sensor_rot_np[env_idx, 3]],  # w
                                                sensor_position,
                                                width=0.02,
                                                length=0.3)

                # Draw rays and hits
                for i in range(ray_caster.num_rays):
                    hit_point = data.ray_hits[env_idx, i].cpu().numpy()
                    hit_found = data.ray_hits_found[env_idx, i].cpu().numpy()

                    if hit_found:
                        # Draw ray hit
                        visualizer.draw_point(env_idx, hit_point, color=(1, 0, 0), size=0.05)
                        # Draw ray (green for hits)
                        visualizer.draw_line(env_idx, [sensor_position, hit_point], color=(0, 1, 0))
                    else:
                        # Compute ray direction and endpoint
                        ray_dir = ray_caster.ray_directions[env_idx, i].cpu().numpy()
                        if ray_caster.cfg.attach_yaw_only:
                            # Apply yaw rotation if needed
                            if device == "cuda:0":
                                sensor_rot_np = sensor_rot.cpu().numpy()
                            yaw = math.atan2(sensor_rot_np[env_idx, 2], sensor_rot_np[env_idx, 3]) * 2
                            rot_matrix = np.array([
                                [math.cos(yaw), -math.sin(yaw), 0],
                                [math.sin(yaw), math.cos(yaw), 0],
                                [0, 0, 1]
                            ])
                            ray_dir = rot_matrix @ ray_dir
                        else:
                            # FIXME
                            # quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
                            # ray_dir = quat_apply(sensor_rot[env_idx], ray_caster.ray_directions[env_idx, i])
                            pass

                        ray_end = sensor_position + ray_dir * ray_caster.cfg.max_distance
                        # Draw ray (blue for misses)
                        visualizer.draw_line(env_idx, [sensor_position, ray_end], color=(0.7, 0.7, 0.7))

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
