from legged_gym.envs.base.legged_robot import LeggedRobot
import torch

def setup_ray_caster(pattern_type, num_rays, angle, mesh_path, device, num_envs):
    """Set up ray caster with specified pattern.

    Note: This function is deprecated. The ray caster is now initialized directly 
    in the LeggedRobotRayCast class.

    Args:
        pattern_type: The type of ray pattern to use (single, grid, cone, spherical, spherical2).
        num_rays: Number of rays to cast.
        angle: Angle for cone pattern.
        mesh_path: Path to terrain mesh file.
        device: PyTorch device to use.
        num_envs: Number of environments.

    Returns:
        A configured RayCaster instance.
    """
    from legged_gym.utils.ray_caster import RayCasterPatternCfg, RayCasterCfg, RayCaster, PatternType

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
            spherical_num_azimuth=8,
            spherical_num_elevation=4
        )
    elif pattern_type == "spherical2":
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.SPHERICAL2,
            spherical2_num_points=32,
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
        max_distance=10.0,
        attach_yaw_only=True
    )
    
    # Create ray caster
    ray_caster = RayCaster(ray_caster_cfg, num_envs, device)
    
    return ray_caster


class LeggedRobotRayCast(LeggedRobot):
    """ Class for legged robots with ray-cast based terrain height measurements.
        Inherits from LeggedRobot class.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Initialize base LeggedRobot class
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # We'll initialize the ray caster after terrain is created, so we just set up
        # configuration attributes here
        self.ray_caster = None
        self.raycast_distances = None
        self.num_ray_observations = 0

        # Now that terrain is created, we can initialize ray casting
        if hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast:
            self._init_ray_caster()

    # def create_sim(self):
    #     """Create simulation, terrain and environments"""
    #     # First create the simulation and terrain using the parent class
    #     super().create_sim()
        
    #     # Now that terrain is created, we can initialize ray casting
    #     if hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast:
    #         self._init_ray_caster()

    def _init_ray_caster(self):
        """Initialize ray caster with configuration from environment settings."""
        from legged_gym.utils.ray_caster import RayCasterPatternCfg, RayCasterCfg, RayCaster, PatternType
        
        # Get ray caster configuration parameters
        pattern_type = self.cfg.raycaster.ray_pattern
        num_rays = self.cfg.raycaster.num_rays
        ray_angle = self.cfg.raycaster.ray_angle
        terrain_file = self.cfg.raycaster.terrain_file
        max_distance = getattr(self.cfg.raycaster, "max_distance", 10.0)
        attach_yaw_only = getattr(self.cfg.raycaster, "attach_yaw_only", False)
        offset_pos = getattr(self.cfg.raycaster, "offset_pos", [0.0, 0.0, 0.0])

        # For spherical pattern, get specific settings if available
        spherical_num_azimuth = getattr(self.cfg.raycaster, "spherical_num_azimuth", 8)
        spherical_num_elevation = getattr(self.cfg.raycaster, "spherical_num_elevation", 4)
        
        # For spherical2 pattern, get specific settings if available
        spherical2_num_points = getattr(self.cfg.raycaster, "spherical2_num_points", 32)
        spherical2_polar_axis = getattr(self.cfg.raycaster, "spherical2_polar_axis", [0.0, 0.0, 1.0])

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
                cone_num_rays=num_rays,
                cone_angle=ray_angle
            )
        elif pattern_type == "spherical":
            pattern_cfg = RayCasterPatternCfg(
                pattern_type=PatternType.SPHERICAL,
                spherical_num_azimuth=spherical_num_azimuth,
                spherical_num_elevation=spherical_num_elevation
            )
        elif pattern_type == "spherical2":
            pattern_cfg = RayCasterPatternCfg(
                pattern_type=PatternType.SPHERICAL2,
                spherical2_num_points=spherical2_num_points,
                spherical2_polar_axis=spherical2_polar_axis
            )
        else:
            print(f"Unknown pattern type: {pattern_type}. Using cone pattern.")
            pattern_cfg = RayCasterPatternCfg(
                pattern_type=PatternType.CONE,
                cone_num_rays=num_rays,
                cone_angle=ray_angle
            )

        # Create ray caster configuration with appropriate terrain data
        ray_caster_cfg = RayCasterCfg(
            pattern_cfg=pattern_cfg,
            max_distance=max_distance,
            offset_pos=offset_pos,
            attach_yaw_only=attach_yaw_only
        )

        # If using terrain objects, use mesh paths
        if self.cfg.terrain.use_terrain_obj and terrain_file:
            ray_caster_cfg.mesh_paths = [terrain_file]
            print(f"Using terrain mesh file for ray casting: {terrain_file}")
        # Otherwise, use the terrain vertices and triangles
        elif hasattr(self, 'terrain') and hasattr(self.terrain, 'vertices') and hasattr(self.terrain, 'triangles'):
            # Convert terrain vertices and triangles to torch tensors if they're not already
            if isinstance(self.terrain.vertices, torch.Tensor):
                ray_caster_cfg.vertices = self.terrain.vertices
            else:
                ray_caster_cfg.vertices = torch.tensor(self.terrain.vertices, device=self.device)
            
            if isinstance(self.terrain.triangles, torch.Tensor):
                ray_caster_cfg.triangles = self.terrain.triangles
            else:
                ray_caster_cfg.triangles = torch.tensor(self.terrain.triangles, device=self.device)
            # Add border_size offset to align with terrain
            if hasattr(self.cfg.terrain, 'border_size'):
                ray_caster_cfg.vertices[:, 0] -= self.cfg.terrain.border_size
                ray_caster_cfg.vertices[:, 1] -= self.cfg.terrain.border_size
            print(f"Using procedurally generated terrain for ray casting with {len(ray_caster_cfg.vertices)} vertices and {len(ray_caster_cfg.triangles)} triangles")
        elif self.cfg.terrain.mesh_type == 'plane':
            import numpy as np
            # Define a large enough ground plane
            size = 100.0
            ray_caster_cfg.vertices = np.array([
                [-size, -size, 0.0],
                [size, -size, 0.0],
                [size, size, 0.0],
                [-size, size, 0.0]
            ], dtype=np.float32)
            ray_caster_cfg.vertices = torch.tensor(ray_caster_cfg.vertices, device=self.device)
            
            # Two triangles to form a rectangle
            ray_caster_cfg.triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ], dtype=np.int32)
            ray_caster_cfg.triangles = torch.tensor(ray_caster_cfg.triangles, device=self.device)
        else:
            raise ValueError("No terrain mesh available for ray casting. Either set use_terrain_obj=True and provide a terrain file, or ensure terrain.vertices and terrain.triangles are available.")

        # Create ray caster
        self.ray_caster = RayCaster(ray_caster_cfg, self.num_envs, self.device)

        # Calculate the observation size including raycast data
        self.num_ray_observations = self.ray_caster.num_rays

        # Initialize raycast points tensor
        self.raycast_distances = torch.zeros(self.num_envs, self.num_ray_observations, device=self.device)
        
        print(f"Ray caster initialized with {pattern_type} pattern, {self.num_ray_observations} rays")

    def _post_physics_step_callback(self):
        """Callback after physics step, updates ray caster with current robot state."""
        super()._post_physics_step_callback()
        if hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast and self.ray_caster is not None:
            # Update ray caster with the current robot positions and orientations
            self.ray_caster.update(
                dt=self.dt,
                sensor_pos=self.base_pos,
                sensor_rot=self.base_quat,
            )
            # Get raycast points as observations
            self.raycast_distances = self._get_raycast_distances()

    def compute_observations(self):
        """Computes observations including raycast data."""
        # Get base observations from parent class
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        # Add height measurements if configured
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # Add raycast data if enabled
        if self.cfg.raycaster.enable_raycast and self.ray_caster is not None and self.raycast_distances is not None:
            # Add normalized raycast distances to the observations
            self.obs_buf = torch.cat((self.obs_buf, self.raycast_distances), dim=-1)

        # Add noise if configured
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_raycast_distances(self, env_ids=None, normalize=True):
        """Get normalized raycast distances as observations.

        Args:
            env_ids: Optional indices of environments to get observations for.

        Returns:
            Tensor of normalized raycast distances for observations.
        """
        # Get data from ray caster
        ray_data = self.ray_caster.data

        # Filter for specific environments if specified
        if env_ids is not None:
            hits = ray_data.ray_hits[env_ids]
            hits_found = ray_data.ray_hits_found[env_ids]
            origins = self.root_states[env_ids, 0:3]
        else:
            hits = ray_data.ray_hits
            hits_found = ray_data.ray_hits_found
            origins = self.root_states[:, 0:3]

        # Calculate distances from ray origins to hit points
        distances = torch.norm(hits - origins.unsqueeze(1), dim=2)
        if not normalize:
            return distances

        # Normalize distances by max distance and invert (closer = higher value)
        max_distance = self.ray_caster.cfg.max_distance
        normalized_distances = 1.0 - torch.clamp(distances / max_distance, 0.0, 1.0)

        # For rays that didn't hit anything, set to 0 (maximum distance)
        normalized_distances = normalized_distances * hits_found.float()

        # Flatten to 1D per environment
        return normalized_distances.reshape(normalized_distances.shape[0], -1)
