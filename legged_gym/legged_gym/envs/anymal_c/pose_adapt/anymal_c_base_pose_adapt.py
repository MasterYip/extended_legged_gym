import os
import torch
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *
from legged_gym.envs.base.base_pose_adapt import BasePoseAdapt
from legged_gym.envs.base.base_pose_adapt_config import BasePoseAdaptCfg, BasePoseAdaptCfgPPO


class AnymalCBasePoseAdaptCfg(BasePoseAdaptCfg):

    # Randomization options
    randomize_init_pos: bool = True
    randomize_init_yaw: bool = True

    class env:
        episode_length_s: float = 20  # Episode length in seconds
        num_envs: int = 4096
        num_observations: int = 132  # Will be updated dynamically based on actual obs
        num_privileged_obs: int = None
        num_actions: int = 6  # linear vel (3) + angular vel (3)
        env_spacing = 1

    class asset(BasePoseAdaptCfg.asset):
        file: str = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c_base.urdf"
        name: str = "anymal_c_base"
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        # Default nominal height for the robot base
        nominal_height: float = 0.3  # Appropriate height for Anymal C [m]


    class init_state(BasePoseAdaptCfg.init_state):
        # Initial pose for the robot
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {}  # empty for base-only robot

    class commands(BasePoseAdaptCfg.commands):
        # Command configuration for AnymalC
        num_commands: int = 3  # lin_x, lin_y, ang_yaw
        resampling_time: float = 4.0  # Time in seconds before resampling commands
        lin_vel_x: list = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y: list = [-1.5, 1.5]  # min max [m/s]
        ang_vel_yaw: list = [-0.5, 0.5]  # min max [rad/s]
        heading_command: bool = False  # Whether to include heading command

    class control(BasePoseAdaptCfg.control):
        decimation: int = 2  # Control frequency decimation
        # PD control parameters for base pose adjustment
        # Medium Tracking
        # position_p_gain: float = 10000.0  # Higher P gain for position
        # position_d_gain: float = 1000.0    # Higher D gain for damping
        # rotation_p_gain: float = 200.0   # Higher P gain for rotation
        # rotation_d_gain: float = 30.0    # Higher D gain for angular damping
        # Fast Tracking
        position_p_gain: float = 90000.0  # Higher P gain for position
        position_d_gain: float = 2000.0    # Higher D gain for damping
        rotation_p_gain: float = 500.0   # Higher P gain for rotation
        rotation_d_gain: float = 40.0    # Higher D gain for angular damping

        # Control method selection
        use_direct_pose_control: bool = False  # If True, uses direct pose setting instead of PD control

    class raycaster(BasePoseAdaptCfg.raycaster):
        # Ray casting parameters
        enable_raycast: bool = True
        ray_pattern: str = "spherical"
        spherical_num_azimuth: int = 16
        spherical_num_elevation: int = 8
        max_distance: float = 2.0
        # Use terrain file or add obstacles programmatically
        terrain_file: str = "/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"
        offset_pos: list = (0.0, 0.0, 0.0)
        attach_yaw_only: bool = False

        # Visualization options
        draw_rays: bool = False
        draw_mesh: bool = False
        draw_hits: bool = False

    class domain_rand(BasePoseAdaptCfg.domain_rand):
        randomize_friction: bool = True
        friction_range: list = [0.5, 1.25]
        randomize_base_mass: bool = False
        added_mass_range: list = [-1., 1.]

    class rewards(BasePoseAdaptCfg.rewards):
        # Penalty weights
        collision_penalty: float = 3.0
        terrain_conformity_penalty: float = 0.0  # Penalty for base not conforming to terrain
        orientation_penalty: float = 0.002  # Small weight for penalizing non-flat orientation

        # Command tracking rewards
        lin_vel_tracking: float = 0.01
        ang_vel_tracking: float = 0.005

        # Downward velocity reward
        downward_vel_reward: float = 0.003  # Higher weight for encouraging downward movement
        downward_vel_scale: float = 0.3  # Scale factor for exponential reward function

        # Termination thresholds
        max_contact_force: float = 500.0  # N

    class obstacles:
        enable: bool = False
        num_obstacles: int = 4
        size_range: list = (0.1, 0.5)
        placement_radius: float = 3.0


class AnymalCBasePoseAdaptCfgPPO(BasePoseAdaptCfgPPO):
    """PPO configuration parameters specifically for AnymalC base pose adapt task."""

    seed: int = 1
    runner_class_name = "OnPolicyRunner"

    class algorithm(BasePoseAdaptCfgPPO.algorithm):
        # Additional/overridden PPO algorithm parameters for AnymalC
        entropy_coef: float = 0.005  # Lower entropy for more focused exploration
        num_learning_epochs: int = 8  # More epochs for better convergence with complex terrain

    class runner(BasePoseAdaptCfgPPO.runner):
        # Runner parameters specific to AnymalC
        max_iterations: int = 2000  # More iterations for this complex task
        save_interval: int = 50  # Save model interval (in iterations)
        experiment_name: str = "anymal_c_base_pose_adapt"
        run_name: str = ""
        multi_stage_rewards: bool = False  # Enable/disable multi-stage rewards

    class policy(BasePoseAdaptCfgPPO.policy):
        # Policy architecture parameters
        init_noise_std: float = 0.8  # Initial action noise std
        actor_hidden_dims: list = [256, 128, 64]  # Larger network for complex task
        critic_hidden_dims: list = [256, 128, 64]  # Larger network for complex task
        activation: str = "elu"  # Hidden layer activation


class AnymalCBasePoseAdapt(BasePoseAdapt):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Initialize the base class
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Set Anymal C specific parameters - nominal_height is now from config
        self.base_index = 0  # Will be properly set in _create_envs

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _create_obstacles(self):
        """Create obstacles in the environment."""
        if not self.cfg.obstacles.enable:
            return

        # Create a box asset for obstacles
        box_options = gymapi.AssetOptions()
        box_options.fix_base_link = True

        # Create obstacles of different sizes
        num_obstacles = self.cfg.obstacles.num_obstacles
        size_range = self.cfg.obstacles.size_range
        placement_radius = self.cfg.obstacles.placement_radius

        for i in range(self.num_envs):
            for _ in range(num_obstacles):
                # Random size for this obstacle
                size = torch.rand(1, device=self.device).item() * (size_range[1] - size_range[0]) + size_range[0]

                # Create box asset for this size
                box_asset = self.gym.create_box(self.sim, size, size, size, box_options)

                # Random position around the robot (cylindrical coordinates)
                angle = torch.rand(1, device=self.device).item() * 2 * np.pi
                radius = torch.rand(1, device=self.device).item() * placement_radius + 1.0  # Minimum 1m from center

                # Convert to Cartesian
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = size / 2  # Half height so it sits on the ground

                # Set obstacle pose
                obstacle_pose = gymapi.Transform()
                obstacle_pose.p = gymapi.Vec3(x, y, z)

                # Create obstacle
                obstacle_handle = self.gym.create_actor(self.envs[i], box_asset, obstacle_pose, f"obstacle_{i}_{_}", i, 0)

                # Set obstacle color
                self.gym.set_rigid_body_color(
                    self.envs[i],
                    obstacle_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.1, 0.0)
                )

    def _draw_debug_vis(self):
        """Draw debug visualization for the robot.

        Override to handle the 2D root_states tensor instead of 3D.
        """
        # Clear previous visualizations
        if hasattr(self, 'vis'):
            self.vis.clear()
        else:
            self.gym.clear_lines(self.viewer)

        # Draw base velocity and command visualization
        # Fix for 2D root_states tensor [num_envs, 13] instead of 3D
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        cmd_vel_world = quat_rotate(self.base_quat, self.commands[:, :3]).cpu().numpy()
        cmd_vel_world[:, 2] = 0.0

        # For each environment, draw ray visualization
        if hasattr(self, 'ray_caster') and self.cfg.raycaster.enable_raycast:
            ray_data = self.ray_caster.data
            ray_dir_length = self.cfg.raycaster.max_distance  # Length of ray visualization

            # Draw mesh for debugging (only once)
            if hasattr(self, 'vis') and not self._mesh_drawn and self.cfg.raycaster.draw_mesh:
                # Draw WARP terrain mesh for debugging
                for mesh_path, warp_mesh in self.ray_caster.meshes.items():
                    # Get mesh vertices from WARP
                    if hasattr(warp_mesh, 'points'):
                        # Convert WARP mesh vertices to numpy
                        vertices = warp_mesh.points.numpy()
                        indices = warp_mesh.indices.numpy().reshape(-1, 3)

                        # Calculate and print WARP mesh bounds and center
                        min_bounds = np.min(vertices, axis=0)
                        max_bounds = np.max(vertices, axis=0)
                        warp_center = (min_bounds + max_bounds) / 2
                        print(f"WARP mesh path: {mesh_path}")
                        print(f"WARP mesh bounds: [{min_bounds}, {max_bounds}]")
                        print(f"WARP mesh center: {warp_center}")

                        # Draw WARP mesh center with a different color
                        self.vis.draw_point(0, warp_center, color=(1, 0, 1), size=0.1)

                        # Draw a subset of the edges to visualize the mesh structure
                        edge_step = max(1, len(indices) // 5000)  # Limit to ~5k edges for performance
                        for j in range(0, len(indices), edge_step):
                            face = indices[j]
                            # Draw the three edges of this triangle
                            v1 = vertices[face[0]]
                            v2 = vertices[face[1]]
                            v3 = vertices[face[2]]
                            self.vis.draw_line(0, [v1, v2], color=(0.5, 0.5, 1.0))
                            self.vis.draw_line(0, [v2, v3], color=(0.5, 0.5, 1.0))
                            self.vis.draw_line(0, [v3, v1], color=(0.5, 0.5, 1.0))

                # Mark mesh as drawn so we only do this expensive operation once
                # self._mesh_drawn = True

            # Draw ray visualization for each environment
            for i in range(self.num_envs):
                # Get base position
                base_pos = self.base_pos[i].cpu().numpy()

                # Draw target coordinate frame at robot position
                if hasattr(self, 'vis'):
                    # Draw base position and coordinate frame
                    quat_np = self.target_quat[i].cpu().numpy()
                    pos = self.target_pos[i].cpu()
                    self.vis.draw_frame_from_quat(
                        i,
                        [quat_np[0], quat_np[1], quat_np[2], quat_np[3]],
                        pos,
                        width=0.02,
                        length=0.2
                    )

                # Draw base velocity vector
                self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0), width=0.01)

                # Draw target velocity vector
                target_pos = self.target_pos[i].cpu().numpy()
                self.vis.draw_arrow(i, target_pos, target_pos + cmd_vel_world[i], color=(1, 0, 0), width=0.01)

                # Draw rays if enabled
                if self.cfg.raycaster.draw_rays:
                    # Get ray directions and origins
                    ray_dir = self.ray_caster.ray_directions[i, :]
                    quat = self.base_quat[i].repeat(self.ray_caster.num_rays, 1)

                    # Calculate ray origin position with offset
                    if hasattr(self.cfg.raycaster, 'offset_pos'):
                        offset = quat_rotate(quat[0:1], torch.tensor(
                            self.cfg.raycaster.offset_pos, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
                        ray_origin = base_pos + offset
                    else:
                        ray_origin = base_pos

                    # Rotate ray directions to world frame
                    world_dir = quat_rotate(quat, ray_dir).cpu().numpy()

                    # Draw each ray
                    for j in range(self.ray_caster.num_rays):
                        ray_end = ray_origin + world_dir[j] * ray_dir_length

                        # Draw the ray using the visualizer if available
                        if hasattr(self, 'vis'):
                            self.vis.draw_line(i, [ray_origin, ray_end], color=(0.7, 0.7, 0.7))
                        else:
                            # Draw with basic line function if visualizer not available
                            self.gym.add_lines(
                                self.viewer,
                                self.envs[i],
                                1,
                                [
                                    ray_origin[0], ray_origin[1], ray_origin[2],
                                    ray_end[0], ray_end[1], ray_end[2]
                                ],
                                [0.7, 0.7, 0.7]
                            )

            # Draw hit points if enabled
            if self.cfg.raycaster.draw_hits:
                hit_mask = ray_data.ray_hits_found.view(self.num_envs, -1)

                # For each environment with hits
                for i in range(self.num_envs):
                    # Get hit points for this environment that actually had hits
                    env_hit_indices = hit_mask[i].nonzero(as_tuple=True)[0]

                    if len(env_hit_indices) > 0:
                        env_hits = ray_data.ray_hits[i, env_hit_indices].cpu().numpy()

                        # Draw each hit point
                        for hit_pos in env_hits:
                            if hasattr(self, 'vis'):
                                # Use visualizer to draw points
                                self.vis.draw_point(i, hit_pos, color=(1, 0, 0), size=0.05)
                            else:
                                # Draw point as a small sphere (approximated by lines forming a cross)
                                point_size = 0.05
                                x, y, z = hit_pos

                                # Draw X-oriented line
                                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                                   [x-point_size, y, z, x+point_size, y, z],
                                                   [1.0, 0.0, 0.0])  # Red

                                # Draw Y-oriented line
                                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                                   [x, y-point_size, z, x, y+point_size, z],
                                                   [1.0, 0.0, 0.0])  # Red

                                # Draw Z-oriented line
                                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                                   [x, y, z-point_size, x, y, z+point_size],
                                                   [1.0, 0.0, 0.0])  # Red

        return
