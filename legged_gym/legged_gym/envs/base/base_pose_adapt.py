import torch
import numpy as np
import os
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.ray_caster import RayCaster, RayCasterCfg, RayCasterPatternCfg, PatternType
from legged_gym.utils.gym_visualizer import GymVisualizer
from legged_gym.utils.terrain_obj import TerrainObj
from .base_pose_adapt_config import BasePoseAdaptCfg
from legged_gym import LEGGED_GYM_ROOT_DIR


class BasePoseAdapt(BaseTask):
    def __init__(self, cfg: BasePoseAdaptCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.dt = sim_params.dt
        # NOTE: this dt is sim dt, not control dt (different from legged_robot.py)

        # Base parameters - to be set by derived classes
        self.nominal_height = self.cfg.asset.nominal_height
        self.nominal_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim_device)  # Identity quaternion

        # Debug visualization flag
        self.debug_viz = True
        self._mesh_drawn = False  # Track if the mesh has been drawn

        # Initialize the base task
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Setup ray caster
        if self.cfg.raycaster.enable_raycast:
            self._setup_ray_caster()
        self._init_buffers()

        # Initialize visualizer if viewer is available
        if not headless and hasattr(self, 'viewer'):
            self.vis = GymVisualizer(self.gym, self.sim, self.viewer, self.envs)

    # Env Setup
    def _setup_ray_caster(self):
        """Set up the ray caster with the specified pattern."""
        pattern_type = self.cfg.raycaster.ray_pattern

        # Create pattern configuration based on pattern type
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
                cone_num_rays=self.cfg.raycaster.num_rays,
                cone_angle=self.cfg.raycaster.ray_angle
            )
        elif pattern_type == "spherical":
            pattern_cfg = RayCasterPatternCfg(
                pattern_type=PatternType.SPHERICAL,
                spherical_num_azimuth=self.cfg.raycaster.spherical_num_azimuth,
                spherical_num_elevation=self.cfg.raycaster.spherical_num_elevation
            )
        elif pattern_type == "spherical2":
            pattern_cfg = RayCasterPatternCfg(
                pattern_type=PatternType.SPHERICAL2,
                spherical2_num_points=self.cfg.raycaster.spherical2_num_points,
                spherical2_polar_axis=self.cfg.raycaster.spherical2_polar_axis
            )
        else:
            print(f"Unknown pattern type: {pattern_type}. Using spherical pattern.")
            pattern_cfg = RayCasterPatternCfg(
                pattern_type=PatternType.SPHERICAL,
                spherical_num_azimuth=self.cfg.raycaster.spherical_num_azimuth,
                spherical_num_elevation=self.cfg.raycaster.spherical_num_elevation
            )

        # Create ray caster configuration
        ray_caster_cfg = RayCasterCfg(
            pattern_cfg=pattern_cfg,
            mesh_paths=[self.cfg.raycaster.terrain_file] if self.cfg.raycaster.terrain_file else [],
            max_distance=self.cfg.raycaster.max_distance,
            offset_pos=self.cfg.raycaster.offset_pos,
            attach_yaw_only=self.cfg.raycaster.attach_yaw_only
        )

        # Create ray caster
        self.ray_caster = RayCaster(ray_caster_cfg, self.num_envs, self.device)
        self.num_ray_observations = self.ray_caster.num_rays

        # Update observation size based on ray count
        self.num_obs = self.num_ray_observations + 5 + self.cfg.commands.num_commands
        self.cfg.env.num_observations = self.num_obs

    def _init_buffers(self):
        """Initialize buffers for simulation."""
        # Get actor root state
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        # Get rigid body state
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)

        # Get contact force tensor
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        self.contact_forces = self.contact_forces.view(self.num_envs, -1, 3)

        # Base state
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # linear velocity in base frame
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Target state
        self.target_pos = self.base_pos
        self.target_quat = self.base_quat
        self.target_quat[:, 3] = 1.0  # Set to identity quaternion

        # Command buffers
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # lin_vel_x, lin_vel_y, ang_vel_yaw
        self.commands_scale = torch.tensor([1.0, 1.0, 1.0],
                                           device=self.device, requires_grad=False)[:self.cfg.commands.num_commands]

        # Command resampling timer
        self.command_resample_time = self.cfg.commands.resampling_time / self.dt
        self.command_timer = torch.ones(self.num_envs, device=self.device, dtype=torch.float) * self.command_resample_time

        # Ray cast distances
        if self.cfg.raycaster.enable_raycast:
            self.raycast_distances = torch.zeros(self.num_envs, self.num_ray_observations, device=self.device)

        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_episode_length = int(self.cfg.env.episode_length_s / (self.dt * self.cfg.control.decimation))

        # Target velocities from policy output
        self.target_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Reward component tracking
        self.episode_sums = {
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "lin_vel_tracking_reward": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_tracking_reward": torch.zeros(self.num_envs, device=self.device),
            "downward_reward": torch.zeros(self.num_envs, device=self.device),
            "nominal_alignment_reward": torch.zeros(self.num_envs, device=self.device),
            "collision_penalty": torch.zeros(self.num_envs, device=self.device),
            "close_distance_penalty": torch.zeros(self.num_envs, device=self.device),
            "terrain_conformity_penalty": torch.zeros(self.num_envs, device=self.device),
            "orientation_penalty": torch.zeros(self.num_envs, device=self.device),
        }

        # Obs and action buffers - updated to match our observation size
        # Observations: raycast distances + height diff (1) + quat diff (4) + commands (3)
        if self.cfg.raycaster.enable_raycast:
            self.num_obs = self.num_ray_observations + 5 + self.cfg.commands.num_commands
        else:
            self.num_obs = 5 + self.cfg.commands.num_commands
        self.cfg.env.num_observations = self.num_obs

        # Actions: linear vel (3) + angular vel (3)
        # FIXME: should not override
        self.cfg.env.num_actions = 6

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.action_buf = torch.zeros(self.num_envs, self.cfg.env.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.action_buf)

        # Initialize extras dictionary
        self.extras = {}

    def _setup_enhanced_lighting(self):
        """Setup enhanced lighting for better visibility in confined spaces."""
        # Add ambient light
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(1, 1, 1))

        # Add multiple directional lights for better coverage
        # Main directional light from above
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(0.7, 0.7, 0.7), gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(1, 1, -1))

        # Side light to illuminate between layers
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(0.5, 0.5, 0.5),
                                      gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(-1, 1, -0.5))

        # Fill light from opposite side
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(0.5, 0.5, 0.5),
                                      gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1, -1, -0.5))

    def create_sim(self):
        """Create simulation, terrain, and environments."""
        self.up_axis_idx = 2  # z-up
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._setup_enhanced_lighting()

        # Create terrain
        if hasattr(self.cfg, 'terrain') and self.cfg.terrain.use_terrain_obj:
            self._create_terrain_with_obj()
        else:
            # Create ground plane
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = 1.0
            plane_params.dynamic_friction = 1.0
            plane_params.restitution = 0.0
            self.gym.add_ground(self.sim, plane_params)

        # Create robot environments first
        self._create_envs()

        # Create obstacles if specified - AFTER environments are created
        if hasattr(self.cfg, 'obstacles') and self.cfg.obstacles.enable:
            self._create_obstacles()

    def _create_terrain_with_obj(self):
        """Create terrain using TerrainObj class."""
        # Initialize terrain object
        self.terrain = TerrainObj(self.cfg.terrain, verbose=False)

        # Setup terrain parameters
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        # Add terrain mesh to simulation
        vertices = self.terrain.vertices.flatten()
        triangles = self.terrain.triangles.flatten()
        self.gym.add_triangle_mesh(
            self.sim,
            vertices.astype(np.float32),
            triangles.astype(np.uint32),
            tm_params
        )

        # Update ray caster terrain file if not set
        if self.cfg.raycaster.enable_raycast and not self.cfg.raycaster.terrain_file and self.cfg.terrain.terrain_file:
            self.cfg.raycaster.terrain_file = self.cfg.terrain.terrain_file

    def _create_obstacles(self):
        """Create obstacles in the environment."""
        # Implementation depends on specific obstacle configuration
        pass

    def _create_envs(self):
        """Create environments with Anymal C robots."""

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose,
                                                 self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _get_env_origins(self):
        """ Sets environment origins with proper heights above terrain using raycast.
            For trimesh terrain, evaluates actual terrain height at each point.
            Otherwise creates a grid at nominal height.
        """
        if self.cfg.terrain.mesh_type == "trimesh":
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            if hasattr(self.cfg.terrain, 'random_origins') and self.cfg.terrain.random_origins:
                # Random origin generation for multi-layer environments
                print("Using random origins generation for multi-layer environment")
                valid_origins_count = 0
                attempts = 0
                max_attempts = self.cfg.terrain.origin_generation_max_attempts
                required_clearance = self.nominal_height * self.cfg.terrain.height_clearance_factor

                # Arrays to store valid positions
                valid_positions = []

                # Define the range for random position sampling
                x_min, x_max = self.cfg.terrain.origins_x_range
                y_min, y_max = self.cfg.terrain.origins_y_range

                print(f"Generating random origins with clearance: {required_clearance:.2f} m")
                print(f"Position range X: [{x_min}, {x_max}], Y: [{y_min}, {y_max}]")

                # Store bounds for logging
                min_ground_height = float('inf')
                max_ground_height = float('-inf')
                min_ceiling_height = float('inf')
                max_ceiling_height = float('-inf')

                # Generate valid positions with sufficient height clearance
                while valid_origins_count < self.num_envs and attempts < max_attempts:
                    # Generate batch of random positions
                    batch_size = min(1000, max_attempts - attempts)
                    if batch_size <= 0:
                        break

                    # Generate random XY positions
                    rand_x = torch.rand(batch_size, device=self.device) * (x_max - x_min) + x_min
                    rand_y = torch.rand(batch_size, device=self.device) * (y_max - y_min) + y_min
                    positions = torch.stack([rand_x, rand_y], dim=1)

                    # Update attempt counter
                    attempts += batch_size

                    if hasattr(self, 'terrain') and hasattr(self.terrain, 'get_heights_batch'):
                        try:
                            # Get ground heights by casting rays downward
                            ground_heights = self.terrain.get_heights_batch(positions.cpu(), max_height=20.0, cast_dir=1)
                            ground_heights = torch.from_numpy(ground_heights).to(self.device)

                            # Get ceiling heights by casting rays upward
                            ceiling_heights = self.terrain.get_heights_batch(positions.cpu(), max_height=20.0, cast_dir=-1)
                            ceiling_heights = torch.from_numpy(ceiling_heights).to(self.device)

                            # Calculate clearance between ceiling and ground
                            clearance = ceiling_heights - ground_heights

                            # Update min/max bounds for logging
                            min_ground_height = min(min_ground_height, ground_heights.min().item())
                            max_ground_height = max(max_ground_height, ground_heights.max().item())
                            if ceiling_heights.numel() > 0:
                                min_ceiling_height = min(min_ceiling_height, ceiling_heights.min().item())
                                max_ceiling_height = max(max_ceiling_height, ceiling_heights.max().item())

                            # Find positions with sufficient clearance
                            valid_indices = (clearance > required_clearance).nonzero(as_tuple=True)[0]

                            if len(valid_indices) > 0:
                                for idx in valid_indices:
                                    if valid_origins_count < self.num_envs:
                                        # Store valid position with ground height
                                        valid_positions.append((
                                            positions[idx, 0].item(),
                                            positions[idx, 1].item(),
                                            ground_heights[idx].item() + self.nominal_height
                                        ))
                                        valid_origins_count += 1
                                    else:
                                        break
                        except Exception as e:
                            print(f"Error getting terrain heights: {e}")
                            import traceback
                            traceback.print_exc()

                    # Print progress occasionally
                    if attempts % 5000 == 0 or valid_origins_count >= self.num_envs:
                        print(
                            f"Origin generation: {valid_origins_count}/{self.num_envs} valid positions found after {attempts} attempts")
                        print(f"Height ranges - Ground: [{min_ground_height:.2f}, {max_ground_height:.2f}], "
                              f"Ceiling: [{min_ceiling_height:.2f}, {max_ceiling_height:.2f}]")

                if valid_origins_count < self.num_envs:
                    print(
                        f"Warning: Could only find {valid_origins_count}/{self.num_envs} valid positions after {attempts} attempts")
                    print("Some environments may be placed at fallback positions")

                    # Fill any remaining positions with valid ones (repeating if necessary)
                    if valid_origins_count > 0:
                        for i in range(valid_origins_count, self.num_envs):
                            idx = i % valid_origins_count  # Wrap around to reuse valid positions
                            valid_positions.append(valid_positions[idx])
                    else:
                        # If no valid positions found, use a grid as fallback
                        print("No valid positions found, falling back to grid layout")
                        for i in range(self.num_envs):
                            valid_positions.append((
                                (i % 10) * 3.0,  # Simple grid layout
                                (i // 10) * 3.0,
                                self.nominal_height
                            ))

                # Set the environment origins from valid positions
                for i in range(self.num_envs):
                    self.env_origins[i, 0] = valid_positions[i][0]
                    self.env_origins[i, 1] = valid_positions[i][1]
                    self.env_origins[i, 2] = valid_positions[i][2]

                # Log position range of final origins
                print(f"Final environment positions range:")
                print(f"X: [{self.env_origins[:, 0].min().item():.2f}, {self.env_origins[:, 0].max().item():.2f}]")
                print(f"Y: [{self.env_origins[:, 1].min().item():.2f}, {self.env_origins[:, 1].max().item():.2f}]")
                print(f"Z: [{self.env_origins[:, 2].min().item():.2f}, {self.env_origins[:, 2].max().item():.2f}]")

            else:
                # Default grid-based origin generation
                # Create a grid of robots
                num_cols = int(np.floor(np.sqrt(self.num_envs)))
                num_rows = int(np.ceil(self.num_envs / num_cols))
                spacing = self.cfg.env.env_spacing

                # Generate centralized grid coordinates
                # Instead of starting at (0,0), we'll center the grid around (0,0)
                row_indices = torch.arange(num_rows, device=self.device)
                col_indices = torch.arange(num_cols, device=self.device)

                # Calculate offsets to center the grid
                row_offset = (num_rows - 1) / 2
                col_offset = (num_cols - 1) / 2

                # Apply offsets to create centered grid
                centered_rows = (row_indices - row_offset) * spacing
                centered_cols = (col_indices - col_offset) * spacing

                # Create grid using meshgrid
                xx, yy = torch.meshgrid(centered_rows, centered_cols)

                # Assign grid coordinates to env_origins
                self.env_origins[:, 0] = xx.flatten()[:self.num_envs]
                self.env_origins[:, 1] = yy.flatten()[:self.num_envs]

                print(f"Environment grid: {num_rows}x{num_cols}")
                print(
                    f"Environment positions range X: [{self.env_origins[:, 0].min().item():.2f}, {self.env_origins[:, 0].max().item():.2f}]")
                print(
                    f"Environment positions range Y: [{self.env_origins[:, 1].min().item():.2f}, {self.env_origins[:, 1].max().item():.2f}]")

                # Evaluate terrain heights at grid positions
                if hasattr(self, 'terrain') and hasattr(self.terrain, 'get_heights_batch'):
                    try:
                        # Get heights from terrain using raycast
                        heights = self.terrain.get_heights_batch(self.env_origins[:, :2].cpu(), max_height=10.0, cast_dir=-1)
                        self.env_origins[:, 2] = torch.from_numpy(heights).to(self.device) + self.nominal_height
                        print(f"Heights range: min={heights.min():.2f}, max={heights.max():.2f}")
                    except Exception as e:
                        print(f"Error getting terrain heights: {e}")
                        import traceback
                        traceback.print_exc()
                        self.env_origins[:, 2] = self.nominal_height
                else:
                    # Fallback to nominal height if terrain height evaluation is not available
                    print("Warning: Terrain height evaluation not available")
                    self.env_origins[:, 2] = self.nominal_height

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            # Create a grid of robots at nominal height with centering
            num_cols = int(np.floor(np.sqrt(self.num_envs)))
            num_rows = int(np.ceil(self.num_envs / num_cols))
            spacing = self.cfg.env.env_spacing

            # Generate centralized grid coordinates
            row_indices = torch.arange(num_rows, device=self.device)
            col_indices = torch.arange(num_cols, device=self.device)

            # Calculate offsets to center the grid
            row_offset = (num_rows - 1) / 2
            col_offset = (num_cols - 1) / 2

            # Apply offsets to create centered grid
            centered_rows = (row_indices - row_offset) * spacing
            centered_cols = (col_indices - col_offset) * spacing

            # Create grid using meshgrid
            xx, yy = torch.meshgrid(centered_rows, centered_cols)

            # Assign grid coordinates
            self.env_origins[:, 0] = xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = self.nominal_height

    # Env Step
    def step(self, actions):
        """Apply actions, simulate, and compute observations and rewards."""
        # Store and clip actions
        self.last_actions = self.action_buf.clone()
        self.action_buf = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # Resample commands at regular intervals
        self.command_timer += 1
        resampling_indices = self.command_timer >= self.command_resample_time
        if resampling_indices.sum() > 0:
            self.command_timer[resampling_indices] = 0
            self._resample_commands(resampling_indices)

        # Apply actions to adjust the base pose
        self._apply_actions()

        # Simulate physics
        for _ in range(self.cfg.control.decimation):
            self.render()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

        # Update state and compute observations and rewards
        self._update_state()
        self.compute_observations()
        self.compute_reward()

        # Check for termination conditions
        self.check_termination()

        # Reset environments if needed
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # Draw debug visualization if enabled
        if self.viewer and self.debug_viz:
            self._draw_debug_vis()

        # Add episode statistics to extras
        if 'episode' not in self.extras:
            self.extras['episode'] = {}
        self.extras['episode']['avg_episode_len'] = self.episode_length_buf.float().mean().item()
        self.extras['episode']['num_resets'] = len(env_ids)

        # Add raycast statistics
        if self.cfg.raycaster.enable_raycast and self.raycast_distances.numel() > 0:
            min_dist = torch.min(self.raycast_distances).item() if self.raycast_distances.numel() > 0 else 0
            self.extras['episode']['min_raycast_dist'] = min_dist * self.cfg.raycaster.max_distance
            self.extras['episode']['avg_raycast_dist'] = torch.mean(
                self.raycast_distances).item() * self.cfg.raycaster.max_distance

        # Add command statistics
        if self.commands.numel() > 0:
            self.extras['episode']['lin_vel_x_cmd'] = torch.mean(self.commands[:, 0]).item()
            self.extras['episode']['lin_vel_y_cmd'] = torch.mean(
                self.commands[:, 1]).item() if self.cfg.commands.num_commands > 1 else 0.0
            self.extras['episode']['ang_vel_yaw_cmd'] = torch.mean(
                self.commands[:, 2]).item() if self.cfg.commands.num_commands > 2 else 0.0

        # Add actual velocity statistics
        self.extras['episode']['lin_vel_x'] = torch.mean(self.base_lin_vel[:, 0]).item()
        self.extras['episode']['lin_vel_y'] = torch.mean(self.base_lin_vel[:, 1]).item()
        self.extras['episode']['lin_vel_z'] = torch.mean(self.base_lin_vel[:, 2]).item()
        self.extras['episode']['ang_vel_yaw'] = torch.mean(self.base_ang_vel[:, 2]).item()

        # Return observation, reward, reset, and info buffers
        obs_buf = torch.clamp(self.obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _apply_actions(self):
        """Apply the policy actions to the robot's base pose."""
        # Extract velocity commands from action buffer and apply action scaling
        cmd_vel_base = self.action_buf[:, :3] * self.cfg.control.action_scale  # Linear velocity in 3D space
        cmd_ang_base = self.action_buf[:, 3:6] * self.cfg.control.action_scale  # Angular velocity in 3D space

        # Integrate angular velocity to get orientation change
        angle = torch.norm(cmd_ang_base, dim=1) * self.dt * self.cfg.control.decimation
        orientation_delta = quat_from_angle_axis(angle, cmd_ang_base)

        cmd_vel_world = quat_rotate(self.target_quat, cmd_vel_base)

        # Integrate velocities to get target position and orientation
        self.target_pos = self.target_pos + cmd_vel_world * self.dt * self.cfg.control.decimation
        self.target_quat = quat_mul(self.target_quat, orientation_delta)

        # Store target velocities (for reward computation)
        # self.target_lin_vel = cmd_vel_base
        # self.target_ang_vel = cmd_ang_base
        # TODO：Temp set to zero
        self.target_lin_vel = torch.zeros_like(cmd_vel_base)
        self.target_ang_vel = torch.zeros_like(cmd_ang_base)

        # Choose between PD control and direct pose setting based on configuration
        if hasattr(self.cfg.control, 'use_direct_pose_control') and self.cfg.control.use_direct_pose_control:
            # Apply direct pose setting for faster and more stable tracking
            self._apply_pose_target()
        else:
            # Apply PD control (physics-based approach) for more realistic movement
            self._apply_pd_control()

    def _apply_pd_control(self):
        """Apply PD control to move the base to the target pose."""

        # Position Ctrl
        pos_error = self.target_pos - self.base_pos
        vel_error = self.target_lin_vel - self.base_lin_vel
        vel_error_world = quat_rotate(self.base_quat, vel_error)
        # Use a more aggressive P gain and a properly damped D gain
        pos_force = self.cfg.control.position_p_gain * pos_error + self.cfg.control.position_d_gain*vel_error_world

        # Limit maximum force to avoid instability
        # FIXME: This param should be in the config file
        max_force = 500000.0  # Adjust based on robot mass
        pos_force_norm = torch.norm(pos_force, dim=1, keepdim=True)
        force_scale = torch.clamp(max_force / (pos_force_norm + 1e-6), max=1.0)
        pos_force = pos_force * force_scale.repeat(1, 3)

        # Rotation Ctrl
        quat_error = quat_mul(self.target_quat, quat_conjugate(self.base_quat))
        w_error = quat_error[:, 3]
        xyz_error = quat_error[:, 0:3]
        angle = 2 * torch.acos(torch.clamp(w_error, -1.0, 1.0))
        axis = torch.zeros_like(xyz_error)

        mask = angle > 0.01  # Small threshold for numerical stability
        axis[mask] = xyz_error[mask] / torch.sin(angle[mask] / 2).unsqueeze(-1)
        rot_error = axis * angle.unsqueeze(-1)  # Direction is axis, magnitude is angle

        angvel_error = self.target_ang_vel - self.base_ang_vel  # Error in base frame
        angvel_error_world = quat_rotate(self.base_quat, angvel_error)  # Convert to world frame
        rot_force = (self.cfg.control.rotation_p_gain * rot_error +
                     self.cfg.control.rotation_d_gain * angvel_error_world)

        max_torque = 100.0  # Maximum allowed torque magnitude
        rot_force_norm = torch.norm(rot_force, dim=1, keepdim=True)
        torque_scale = torch.clamp(max_torque / (rot_force_norm + 1e-6), max=1.0)
        rot_force = rot_force * torque_scale.repeat(1, 3)

        # Apply
        # Get total number of rigid bodies in the simulation
        total_bodies = self.rigid_body_states.shape[0]
        num_bodies_per_env = total_bodies // self.num_envs

        # Create full force and torque tensors initialized to zero
        full_forces = torch.zeros((total_bodies, 3), device=self.device)
        full_torques = torch.zeros((total_bodies, 3), device=self.device)

        # Calculate indices for the base rigid body of each environment
        base_indices = torch.arange(0, total_bodies, num_bodies_per_env, device=self.device, dtype=torch.long)

        # Set the forces and torques for the base rigid body of each environment
        full_forces[base_indices] = pos_force
        full_torques[base_indices] = rot_force
        # TODO: temp zero
        # full_forces[base_indices] = torch.zeros_like(pos_force)
        # full_torques[base_indices] = torch.zeros_like(rot_force)

        # Apply forces and torques to all rigid bodies
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(full_forces),
            gymtorch.unwrap_tensor(full_torques),
            # gymapi.ENV_SPACE, # FIXME
            gymapi.GLOBAL_SPACE
        )

    def _apply_pose_target(self):
        """Directly set the robot pose to the target pose for faster and more stable tracking.

        This method bypasses the physics simulation for the robot base by directly setting
        the root state of the robot to the target pose. This is useful when precise pose 
        control is needed without the instabilities of PD control.

        Note: This should be used carefully as it can cause unrealistic motion and 
        potential penetration with the environment.
        """
        # Calculate the indices for the root body of each environment
        actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # Create a temporary root state tensor with the target pose
        temp_root_states = self.root_states.clone()

        # Set position to target position
        temp_root_states[:, 0:3] = self.target_pos

        # Set orientation to target orientation
        temp_root_states[:, 3:7] = self.target_quat

        # Set linear velocity based on position difference and time step
        # This helps maintain smooth motion when teleporting
        pos_diff = self.target_pos - self.base_pos
        temp_root_states[:, 7:10] = pos_diff / (self.dt * self.cfg.control.decimation)

        # Set angular velocity to target angular velocity
        temp_root_states[:, 10:13] = self.target_ang_vel

        # Apply the new root states to the simulation
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(temp_root_states),
            gymtorch.unwrap_tensor(actor_indices),
            self.num_envs
        )

        # Update our internal state to match what we just set
        self.base_pos = self.target_pos.clone()
        self.base_quat = self.target_quat.clone()
        self.base_lin_vel = self.target_lin_vel.clone()
        self.base_ang_vel = self.target_ang_vel.clone()

    def _update_state(self):
        """Update the state from the simulation."""
        # Refresh state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Update ray caster if enabled
        if self.cfg.raycaster.enable_raycast:
            self.ray_caster.update(
                dt=self.dt * self.cfg.control.decimation,
                sensor_pos=self.base_pos,
                sensor_rot=self.base_quat
            )
            self.raycast_distances = self._get_raycast_distances()

    def _get_raycast_distances(self):
        """Get normalized raycast distances as observations."""
        # Get data from ray caster
        ray_data = self.ray_caster.data
        hits = ray_data.ray_hits
        hits_found = ray_data.ray_hits_found

        # Calculate distances from ray origins to hit points
        distances = torch.norm(hits - self.base_pos.unsqueeze(1), dim=2)

        # Normalize distances by max distance and invert (closer = higher value)
        max_distance = self.ray_caster.cfg.max_distance
        normalized_distances = 1.0 - torch.clamp(distances / max_distance, 0.0, 1.0)

        # For rays that didn't hit anything, set to 0 (maximum distance)
        normalized_distances = normalized_distances * hits_found.float()

        return normalized_distances

    def compute_observations(self):
        """Compute observations for the policy."""
        # Observations include:
        # 1. Normalized raycast distances
        # 2. Nominal height and current height difference
        # 3. Nominal quaternion and current quaternion difference
        # 4. Command signals

        # Raycast distances
        self.obs_buf[:, :self.num_ray_observations] = self.raycast_distances.reshape(self.num_envs, -1)

        # Height information
        height_diff = self.base_pos[:, 2] - self.nominal_height
        self.obs_buf[:, self.num_ray_observations] = height_diff

        # Quaternion information - difference between current and nominal
        quat_diff = quat_mul(self.base_quat, quat_conjugate(self.nominal_quat.repeat(self.num_envs, 1)))
        self.obs_buf[:, self.num_ray_observations + 1: self.num_ray_observations + 5] = quat_diff

        # Command signals (in local frame)
        self.obs_buf[:, self.num_ray_observations + 5: self.num_ray_observations +
                     5 + self.cfg.commands.num_commands] = self.commands

    # Rewards
    def compute_reward(self):
        """Compute rewards for the current state."""
        # Reset reward buffer
        self.rew_buf[:] = 0.0

        # Reward components - each function now returns the properly scaled value
        collision_penalty = self._reward_collision()
        terrain_conformity_penalty = self._reward_terrain_conformity()
        orientation_penalty = self._reward_orientation()

        # Velocity tracking rewards - already scaled within the function
        lin_vel_reward, ang_vel_reward = self._reward_velocity_tracking(return_separate=True)

        # Downward velocity reward
        downward_reward = self._reward_velocity_downward()

        # Store reward components for tracking
        self.episode_sums["collision_penalty"] += collision_penalty
        self.episode_sums["terrain_conformity_penalty"] += terrain_conformity_penalty
        self.episode_sums["orientation_penalty"] += orientation_penalty
        self.episode_sums["lin_vel_tracking_reward"] += lin_vel_reward
        self.episode_sums["ang_vel_tracking_reward"] += ang_vel_reward
        self.episode_sums["downward_reward"] += downward_reward

        # Combine rewards - no need to apply scaling here anymore
        self.rew_buf = (
            -collision_penalty
            - terrain_conformity_penalty
            - orientation_penalty
            + lin_vel_reward + ang_vel_reward
            + downward_reward
        )

        # Track total reward
        self.episode_sums["total_reward"] += self.rew_buf

        # Add reward information to extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            self.extras['episode'][key] = self.episode_sums[key].mean().item()

    def _reward_close_distance(self):
        """Penalty for being too close to obstacles."""
        # Find the minimum distance to any obstacle
        min_distances = torch.min(1.0 - self.raycast_distances, dim=1)[0]
        # Penalize distances less than the safe threshold
        close_penalty = torch.clamp(
            (self.cfg.rewards.min_safe_distance - min_distances * self.cfg.raycaster.max_distance)
            / self.cfg.rewards.min_safe_distance,
            0.0, 1.0
        )
        return close_penalty

    def _reward_collision(self):
        """Penalty for base collisions with environment."""
        # Sum contact forces on the base
        base_contact_force = torch.norm(self.contact_forces[:, 0, :], dim=1)
        # Normalize by maximum contact force
        normalized_force = torch.clamp(base_contact_force / self.cfg.rewards.max_contact_force, 0.0, 1.0)
        return normalized_force * self.cfg.rewards.collision_penalty

    # TODO: this penalty is not good
    def _reward_terrain_conformity(self):
        """Penalty for base bottom plane not conforming to ground terrain.

        Evaluates how well the base is aligned with the terrain by checking:
        1. Downward rays should be at approximately nominal height from terrain
        2. Other rays should follow distance = nominal_height / cos(theta)
           where theta is the ray cast angle from vertical
        """
        if not hasattr(self, 'ray_caster') or not self.cfg.raycaster.enable_raycast:
            return torch.zeros(self.num_envs, device=self.device)

        # Get ray data
        ray_data = self.ray_caster.data
        ray_directions = self.ray_caster.ray_directions  # [num_envs, num_rays, 3]

        # Calculate dot product of ray directions with downward vector (0,0,-1)
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        # Vectorized conversion of ray directions from local to world space
        # Create repeated base quaternions for all rays [num_envs, num_rays, 4]
        num_rays = ray_directions.shape[1]
        repeated_quats = self.base_quat.unsqueeze(1).repeat(1, num_rays, 1)  # [num_envs, num_rays, 4]

        # Reshape for efficient batch processing
        flat_quats = repeated_quats.reshape(-1, 4)  # [num_envs*num_rays, 4]
        flat_ray_dirs = ray_directions.reshape(-1, 3)  # [num_envs*num_rays, 3]

        # Perform rotation in one batch operation
        flat_world_ray_dirs = quat_rotate(flat_quats, flat_ray_dirs)  # [num_envs*num_rays, 3]

        # Reshape back to original dimensions
        world_ray_dirs = flat_world_ray_dirs.reshape(self.num_envs, num_rays, 3)  # [num_envs, num_rays, 3]

        # Calculate cosine of angles between ray directions and down vector
        cos_angles = (world_ray_dirs @ down_vec).squeeze(-1)  # [num_envs, num_rays]

        # Calculate expected distances based on ray angle: nominal_height / cos(theta)
        # Handle the near-horizontal rays (cos_angle close to 0) to prevent division by zero
        # Clamp cosine values to avoid instability
        cos_angles_clamped = torch.clamp(cos_angles, min=0.1)  # Minimum cos(theta) = 0.1 (~84 degrees)
        expected_distances = self.nominal_height / cos_angles_clamped

        # Limit the maximum expected distance to avoid extremely large values
        max_expected = self.nominal_height * 5.0  # Cap at 5x nominal height
        expected_distances = torch.clamp(expected_distances, max=max_expected)

        # Get the hit points and hits mask
        hits = ray_data.ray_hits  # [num_envs, num_rays, 3]
        hits_found = ray_data.ray_hits_found  # [num_envs, num_rays]

        # Calculate distances from base position to hit points
        # Expand base_pos for broadcasting: [num_envs, 1, 3]
        expanded_base_pos = self.base_pos.unsqueeze(1)

        # Calculate all distances in one vectorized operation
        actual_distances = torch.norm(hits - expanded_base_pos, dim=2)  # [num_envs, num_rays]

        # For rays that didn't hit, set to max distance
        actual_distances = torch.where(
            hits_found,
            actual_distances,
            torch.tensor(self.cfg.raycaster.max_distance, device=self.device)
        )

        # Calculate error between expected and actual distances
        distance_errors = torch.abs(actual_distances - expected_distances)

        # Create weights with higher emphasis on downward rays
        # Map cosine from [-1,1] to [0,1] with higher values for downward rays
        weights = (cos_angles + 1.0) / 2.0
        weights = weights.pow(2)  # Square to emphasize downward rays

        # Only consider rays that hit something for the weighted average
        masked_weights = weights * hits_found.float()

        # Calculate weighted mean error - handle case where no rays hit
        sum_weights = masked_weights.sum(dim=1)
        valid_envs = sum_weights > 0

        # Initialize penalty tensor
        conformity_penalty = torch.zeros(self.num_envs, device=self.device)

        # Calculate penalty only for valid environments (with ray hits)
        if valid_envs.any():
            # Calculate weighted average of distance errors
            weighted_errors = (distance_errors * masked_weights).sum(dim=1)[valid_envs]
            valid_weights_sum = sum_weights[valid_envs]

            conformity_penalty[valid_envs] = weighted_errors / valid_weights_sum

            # Normalize penalty to [0,1] range
            conformity_penalty[valid_envs] = torch.clamp(
                conformity_penalty[valid_envs] / self.nominal_height,
                0.0,
                1.0
            )

        # Apply scaling here in the reward function
        return conformity_penalty * self.cfg.rewards.terrain_conformity_penalty

    def _reward_orientation(self):
        """Penalize non-flat base orientation.

        Uses the projected gravity vector to measure how much the base deviates from being flat.
        A perfectly flat base would have projected gravity of [0, 0, -1], with x,y components = 0.
        """
        # Sum squared x,y components of the projected gravity (larger when base is tilted)
        orientation_penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

        # Normalize to range [0,1] for consistency with other penalties
        # A tilt of ~30 degrees results in a penalty of ~0.25
        orientation_penalty = torch.clamp(orientation_penalty, 0.0, 1.0)

        # Apply scaling here in the reward function
        return orientation_penalty * self.cfg.rewards.orientation_penalty

    def _reward_velocity_tracking(self, return_separate=False):
        """Reward for tracking the commanded velocities."""
        # Extract command velocities
        cmd_lin_vel_x = self.commands[:, 0]
        cmd_lin_vel_y = self.commands[:, 1] if self.cfg.commands.num_commands > 1 else torch.zeros_like(self.commands[:, 0])
        cmd_ang_vel_yaw = self.commands[:, 2] if self.cfg.commands.num_commands > 2 else torch.zeros_like(self.commands[:, 0])

        # Create command velocity vectors in 3D space
        cmd_lin_vel = torch.zeros_like(self.base_lin_vel)
        cmd_lin_vel[:, 0] = cmd_lin_vel_x
        cmd_lin_vel[:, 1] = cmd_lin_vel_y

        cmd_ang_vel = torch.zeros_like(self.base_ang_vel)
        cmd_ang_vel[:, 2] = cmd_ang_vel_yaw

        # Compute tracking error
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel - cmd_lin_vel), dim=1)
        ang_vel_error = torch.sum(torch.square(self.base_ang_vel - cmd_ang_vel), dim=1)

        # Compute tracking rewards
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25) * self.cfg.rewards.lin_vel_tracking
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25) * self.cfg.rewards.ang_vel_tracking

        if return_separate:
            return lin_vel_reward, ang_vel_reward
        return lin_vel_reward + ang_vel_reward

    def _reward_velocity_downward(self):
        """Reward for moving downward toward the ground.

        This reward encourages the robot to descend toward the ground by 
        rewarding negative vertical velocity (downward movement).
        """
        # Extract vertical velocity component (negative is downward)
        vertical_vel = self.base_lin_vel[:, 2]

        # Reward negative velocity (downward motion)
        # We use an exponential function to reward downward motion more as velocity increases
        # For zero or upward velocity, reward is zero
        # For downward velocity, reward increases with speed
        downward_mask = vertical_vel < 0
        downward_reward = torch.zeros_like(vertical_vel)

        # Apply reward only for downward movement
        if downward_mask.any():
            # Convert to positive numbers for easier calculation
            downward_speed = -vertical_vel[downward_mask]
            # Exponential reward that increases with downward speed
            downward_reward[downward_mask] = 1.0 - torch.exp(-downward_speed / self.cfg.rewards.downward_vel_scale)

        # Scale the reward
        return downward_reward * self.cfg.rewards.downward_vel_reward

    # Reset Env
    def check_termination(self):
        """Check if environments need to be reset."""
        # Reset if base collision force exceeds threshold
        base_contact_force = torch.norm(self.contact_forces[:, 0, :], dim=1)
        self.reset_buf = base_contact_force > self.cfg.rewards.max_contact_force * 2.0

        # Reset on timeout
        self.episode_length_buf += 1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """Reset specific environments."""
        if len(env_ids) == 0:
            return

        # Reset episode length and done flags
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Reset episode reward sums
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0

        # Reset positions and orientations
        self._reset_root_states(env_ids)

        # Resample initial state with some randomization
        self._randomize_base_pose(env_ids)

    def _reset_root_states_AIgen(self, env_ids):
        """Reset root states with initial position and orientation from configuration."""
        # Use initial state configuration
        # Set position and orientation from init_state
        self.root_states[env_ids, 0:3] = self.base_init_state[:3].unsqueeze(0).repeat(len(env_ids), 1)

        # Randomize X-Y position a bit (if desired)
        if hasattr(self.cfg, 'randomize_init_pos') and self.cfg.randomize_init_pos:
            self.root_states[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)

        # Set height based on terrain or fixed value
        if hasattr(self.cfg, 'terrain') and self.cfg.terrain.use_terrain_obj and hasattr(self, 'terrain'):
            # Get height at reset positions
            x = self.root_states[env_ids, 0].cpu().numpy()
            y = self.root_states[env_ids, 1].cpu().numpy()
            heights = torch.tensor(
                [self.terrain.get_height(x[i], y[i]) for i in range(len(env_ids))],
                device=self.device,
                dtype=self.root_states.dtype
            )
            # Set robot height above terrain
            self.root_states[env_ids, 2] = heights + self.nominal_height
        else:
            # Fixed height on flat ground (use configured height)
            self.root_states[env_ids, 2] = self.base_init_state[2]

        # Set orientation from init state
        self.root_states[env_ids, 3:7] = self.base_init_state[3:7].unsqueeze(0).repeat(len(env_ids), 1)

        # Add some randomization to the yaw if desired
        if hasattr(self.cfg, 'randomize_init_yaw') and self.cfg.randomize_init_yaw:
            # Random yaw rotation
            random_yaw = torch_rand_float(-3.14159, 3.14159, (len(env_ids), 1), device=self.device)

            # Create axis vectors with proper shape
            z_axis = torch.zeros((len(env_ids), 3), device=self.device)
            z_axis[:, 2] = 1.0  # Set z-component to 1.0 for all environments

            # Generate quaternion from yaw angle - squeeze random_yaw to remove the second dimension
            quat = quat_from_angle_axis(random_yaw.squeeze(-1), z_axis)

            self.root_states[env_ids, 3:7] = quat

        # Set velocities from initial state
        self.root_states[env_ids, 7:13] = self.base_init_state[7:13].unsqueeze(0).repeat(len(env_ids), 1)

        # Reset target positions to match new root positions
        self.target_pos[env_ids] = self.root_states[env_ids, 0:3]
        self.target_quat[env_ids] = self.root_states[env_ids, 3:7]

        # Set State in gym
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.target_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.target_pos[env_ids] = self.root_states[env_ids, 0:3]
        # base velocities
        # [7:10]: lin vel, [10:13]: ang vel
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _randomize_base_pose(self, env_ids):
        """Add randomization to the base pose for exploration."""
        # Add some random noise to position and orientation for exploration
        pos_noise = torch.rand(len(env_ids), 3, device=self.device) * 0.2 - 0.1
        pos_noise[:, 2] *= 0.1  # Less noise in height

        # Small random rotation
        angle = torch.rand(len(env_ids), device=self.device) * np.pi * 0.2 - np.pi * 0.1
        axis = torch.nn.functional.normalize(torch.rand(len(env_ids), 3, device=self.device), dim=1)

        # Convert to quaternion
        quat_noise = torch.zeros(len(env_ids), 4, device=self.device)
        quat_noise[:, 3] = torch.cos(angle / 2)
        quat_noise[:, 0:3] = axis * torch.sin(angle / 2).unsqueeze(1)

        # Apply the noise to the base pose
        self.root_states[env_ids, 0:3] = self.root_states[env_ids, 0:3] + pos_noise
        self.root_states[env_ids, 3:7] = quat_mul(
            self.root_states[env_ids, 3:7],
            quat_noise
        )

        # Apply the states to the simulation
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _resample_commands(self, env_ids):
        """Resample commands for specified environments."""
        # Sample command values within specified ranges
        if self.cfg.commands.num_commands >= 1:
            self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (
                self.cfg.commands.lin_vel_x[1] - self.cfg.commands.lin_vel_x[0]
            ) + self.cfg.commands.lin_vel_x[0]

        if self.cfg.commands.num_commands >= 2:
            self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (
                self.cfg.commands.lin_vel_y[1] - self.cfg.commands.lin_vel_y[0]
            ) + self.cfg.commands.lin_vel_y[0]

        if self.cfg.commands.num_commands >= 3:
            self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (
                self.cfg.commands.ang_vel_yaw[1] - self.cfg.commands.ang_vel_yaw[0]
            ) + self.cfg.commands.ang_vel_yaw[0]

    # Debug Vis
    def _draw_debug_vis(self):
        """Draw debug visualization for the robot."""
        # Clear previous visualizations
        if hasattr(self, 'vis'):
            self.vis.clear()
        else:
            self.gym.clear_lines(self.viewer)

        # Draw base velocity and command visualization
        lin_vel = self.root_states[:, 0, 7:10].cpu().numpy()
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

                # # Draw coordinate frame at robot position
                # if hasattr(self, 'vis'):
                #     # Draw base position and coordinate frame
                #     quat_np = self.base_quat[i].cpu().numpy()
                #     self.vis.draw_frame_from_quat(
                #         i,
                #         [quat_np[0], quat_np[1], quat_np[2], quat_np[3]],
                #         base_pos,
                #         width=0.02,
                #         length=0.2
                #     )

                # Draw base velocity vector
                self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0), width=0.01)

                # Draw target velocity vector
                target_pos = self.target_pos[i].cpu().numpy()
                self.vis.draw_arrow(i, target_pos, target_pos + cmd_vel_world[i], color=(1, 0, 0), width=0.01)

                # Draw target position
                self.vis.draw_point(i, self.target_pos[i], color=(1, 0, 0), size=0.05)

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
