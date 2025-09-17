import torch
import numpy as np
import trimesh
import warp as wp
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import isaacgym  # Required for isaacgym.torch_utils
from isaacgym.torch_utils import quat_apply
from legged_gym.utils.math_utils import quat_apply_yaw
from isaacgym.torch_utils import quat_apply
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

# disable warp module initialization messages
wp.config.quiet = True
# initialize the warp module
wp.init()

# Utility functions for warp mesh conversion


def convert_to_warp_mesh(vertices, triangles, device="cuda:0"):
    """
    Convert vertices and triangles to a warp mesh.

    Args:
        vertices (numpy.ndarray): Vertices of the mesh.
        triangles (numpy.ndarray): Triangle indices of the mesh.
        device (str): Device to allocate the mesh on.

    Returns:
        wp.Mesh: Warp mesh.
    """
    # For warp, we need to use "cuda" not "cuda:0"
    warp_device = "cpu" if device == "cpu" else "cuda"
    print(f"Creating WARP mesh on device: {warp_device}")

    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=warp_device),
        indices=wp.array(triangles.astype(np.int32).flatten(), dtype=wp.int32, device=warp_device)
    )


@wp.kernel
def raycast_mesh_kernel(
    mesh: wp.uint64,
    ray_origins: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    hits_found: wp.array(dtype=wp.int32),
    max_dist: float,
):
    """WARP kernel for ray casting against a mesh.

    Args:
        mesh: The mesh ID to ray cast against.
        ray_origins: Array of ray origin positions.
        ray_directions: Array of ray direction vectors.
        ray_hits: Output array to store ray hit positions.
        hits_found: Output array to store whether a hit was found (1) or not (0).
        max_dist: Maximum ray cast distance.
    """
    # Get thread ID
    tid = wp.tid()

    # Initialize variables for ray casting
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # Ray cast against the mesh
    hit = wp.mesh_query_ray(
        mesh,
        ray_origins[tid],
        ray_directions[tid],
        max_dist,
        t, u, v, sign, n, f
    )

    # Store results
    if hit:
        # Compute hit position
        ray_hits[tid] = ray_origins[tid] + t * ray_directions[tid]
        hits_found[tid] = 1
    else:
        # Store the ray endpoint at max distance if no hit
        ray_hits[tid] = ray_origins[tid] + ray_directions[tid] * max_dist
        hits_found[tid] = 0


def raycast_mesh(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_dist: float = 100.0,
    mesh: wp.Mesh = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cast rays against a mesh.

    Args:
        ray_origins: Tensor of shape (batch_size, n_rays, 3) or (n_rays, 3)
        ray_directions: Tensor of shape (batch_size, n_rays, 3) or (n_rays, 3)
        max_dist: Maximum distance for ray casting
        mesh: Warp mesh to cast rays against

    Returns:
        Tuple of:
        - hits: Tensor of shape (batch_size, n_rays, 3) or (n_rays, 3)
        - hits_found: Boolean tensor of shape (batch_size, n_rays) or (n_rays)
    """
    if mesh is None:
        raise ValueError("Mesh cannot be None")

    # Check input shapes and flatten if necessary
    input_rank = len(ray_origins.shape)
    if input_rank == 3:
        batch_size, n_rays, _ = ray_origins.shape
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_directions_flat = ray_directions.reshape(-1, 3)
    elif input_rank == 2:
        batch_size, n_rays = 1, ray_origins.shape[0]
        ray_origins_flat = ray_origins
        ray_directions_flat = ray_directions
    else:
        raise ValueError(f"Expected ray_origins to have rank 2 or 3, got {input_rank}")

    # Get device from input tensors
    device = ray_origins.device

    # WARP device is either "cuda" or "cpu", not "cuda:0"
    wp_device = mesh.points.device

    # Create warp arrays
    total_rays = ray_origins_flat.shape[0]
    ray_origins_np = ray_origins_flat.cpu().numpy()
    ray_directions_np = ray_directions_flat.cpu().numpy()

    ray_origins_wp = wp.array(ray_origins_np, dtype=wp.vec3, device=wp_device)
    ray_directions_wp = wp.array(ray_directions_np, dtype=wp.vec3, device=wp_device)
    ray_hits_wp = wp.zeros(total_rays, dtype=wp.vec3, device=wp_device)
    hits_found_wp = wp.zeros(total_rays, dtype=wp.int32, device=wp_device)

    # Launch the kernel
    wp.launch(
        kernel=raycast_mesh_kernel,
        dim=total_rays,
        inputs=[mesh.id, ray_origins_wp, ray_directions_wp, ray_hits_wp, hits_found_wp, float(max_dist)],
        device=wp_device
    )

    # Synchronize to ensure computation is complete
    wp.synchronize()

    # Convert back to torch tensors
    ray_hits = torch.tensor(ray_hits_wp.numpy(), dtype=torch.float32, device=device)
    hits_found = torch.tensor(hits_found_wp.numpy(), dtype=torch.bool, device=device)

    # Reshape back if necessary
    if input_rank == 3:
        ray_hits = ray_hits.reshape(batch_size, n_rays, 3)
        hits_found = hits_found.reshape(batch_size, n_rays)

    return ray_hits, hits_found


class PatternType(Enum):
    """Types of ray casting patterns."""
    SINGLE_RAY = "single_ray"
    GRID = "grid"
    CONE = "cone"
    SPHERICAL = "spherical"
    SPHERICAL2 = "spherical2"  # New uniform spherical pattern


@dataclass
class RayCasterPatternCfg:
    """Configuration for ray casting patterns."""
    pattern_type: PatternType = PatternType.SINGLE_RAY

    # Single ray
    single_ray_direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

    # Grid
    grid_dims: Tuple[int, int] = (5, 5)
    grid_width: float = 1.0
    grid_height: float = 1.0

    # Cone
    cone_num_rays: int = 16
    cone_angle: float = 30.0  # degrees

    # Spherical
    spherical_num_azimuth: int = 8
    spherical_num_elevation: int = 4
    
    # Spherical2 (uniform)
    spherical2_num_points: int = 32
    spherical2_polar_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])  # Direction of polar axis
    ellipsoid_axes: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.3])  # Semi-axes for ellipsoid scaling

    def create_pattern(self, device: str = "cuda:0") -> Tuple[torch.Tensor, torch.Tensor]:
        """Create ray origins and directions according to the pattern.

        Args:
            device: Device to allocate tensors on.

        Returns:
            Tuple of:
            - ray_origins: Tensor of shape (n_rays, 3)
            - ray_directions: Tensor of shape (n_rays, 3)
        """
        if self.pattern_type == PatternType.SINGLE_RAY:
            ray_origins = torch.zeros((1, 3))
            ray_directions = torch.tensor([self.single_ray_direction])

            # Move to device
            ray_origins = ray_origins.to(device)
            ray_directions = ray_directions.to(device)

        elif self.pattern_type == PatternType.GRID:
            rows, cols = self.grid_dims
            ray_origins = torch.zeros((rows * cols, 3))

            # Create grid pattern
            x_vals = torch.linspace(-self.grid_width/2, self.grid_width/2, cols)
            y_vals = torch.linspace(-self.grid_height/2, self.grid_height/2, rows)

            # Move to device
            ray_origins = ray_origins.to(device)
            x_vals = x_vals.to(device)
            y_vals = y_vals.to(device)

            directions = []
            for y in y_vals:
                for x in x_vals:
                    directions.append([1.0, x, y])

            ray_directions = torch.tensor(directions).to(device)
            ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)

        elif self.pattern_type == PatternType.CONE:
            ray_origins = torch.zeros((self.cone_num_rays, 3)).to(device)

            # Create cone pattern
            angle_rad = self.cone_angle * (np.pi / 180)
            angles = torch.linspace(0, 2*np.pi*(1.0-1.0/self.cone_num_rays), self.cone_num_rays).to(device)

            # Cone spreading factor
            spread = torch.sin(torch.tensor(angle_rad).to(device))

            # Create directions
            directions = []
            for angle in angles:
                x = torch.cos(angle) * spread
                y = torch.sin(angle) * spread
                z = torch.cos(torch.tensor(angle_rad).to(device))
                directions.append([z.item(), x.item(), y.item()])  # Forward is z-axis

            ray_directions = torch.tensor(directions).to(device)
            ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)

        elif self.pattern_type == PatternType.SPHERICAL:
            total_rays = self.spherical_num_azimuth * self.spherical_num_elevation
            ray_origins = torch.zeros((total_rays, 3)).to(device)

            # Create spherical pattern
            azimuth_angles = torch.linspace(0, 2*np.pi*(1.0-1.0/self.spherical_num_azimuth),
                                            self.spherical_num_azimuth).to(device)
            elevation_angles = torch.linspace(-np.pi/2, np.pi/2, self.spherical_num_elevation).to(device)

            # Create directions
            directions = []
            for elevation in elevation_angles:
                for azimuth in azimuth_angles:
                    x = torch.cos(elevation) * torch.cos(azimuth)
                    y = torch.cos(elevation) * torch.sin(azimuth)
                    z = torch.sin(elevation)
                    directions.append([x.item(), y.item(), z.item()])

            ray_directions = torch.tensor(directions).to(device)
            
        elif self.pattern_type == PatternType.SPHERICAL2:
            # Create uniform spherical pattern using Fibonacci sphere algorithm
            # This provides approximately uniform distribution of points on a sphere
            ray_origins = torch.zeros((self.spherical2_num_points, 3)).to(device)
            
            # Generate uniform points on a unit sphere using Fibonacci spiral
            golden_ratio = (1 + 5**0.5) / 2
            
            # Pre-allocate direction vectors
            directions = torch.zeros((self.spherical2_num_points, 3), device=device)
            
            # Generate points on unit sphere
            for i in range(self.spherical2_num_points):
                # Fibonacci lattice formula
                y = 1 - (2 * i) / (self.spherical2_num_points - 1)  # y goes from 1 to -1
                radius = (1 - y * y) ** 0.5  # radius at y
                
                # Golden angle increment
                theta = 2 * np.pi * i / golden_ratio
                
                # Convert to Cartesian coordinates
                x = radius * np.cos(theta)
                z = radius * np.sin(theta)
                
                directions[i, 0] = x
                directions[i, 1] = y
                directions[i, 2] = z
            
            # Apply ellipsoid scaling: multiply by semi-axes [a, b, c]
            ellipsoid_axes = torch.tensor(self.ellipsoid_axes, device=device)
            directions = directions * ellipsoid_axes
            
            # Normalize to get unit directions (points on ellipsoid surface projected to unit sphere)
            ray_directions = directions / torch.norm(directions, dim=1, keepdim=True)

            # If a custom polar axis is specified, rotate all directions to align with it
            if not np.allclose(self.spherical2_polar_axis, [0.0, 0.0, 1.0]):
                # Convert polar axis to unit vector
                polar_axis = torch.tensor(self.spherical2_polar_axis, device=device)
                polar_axis = polar_axis / torch.norm(polar_axis)
                
                # Default polar axis is [0,0,1]
                default_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                
                # Find rotation axis and angle
                rotation_axis = torch.cross(default_axis, polar_axis)
                
                # Handle case where vectors are parallel or anti-parallel
                if torch.norm(rotation_axis) < 1e-6:
                    if torch.dot(default_axis, polar_axis) > 0:
                        # Vectors are parallel, no rotation needed
                        pass
                    else:
                        # Vectors are anti-parallel, rotate 180 degrees around any perpendicular axis
                        # Choose x-axis for simplicity
                        rotation_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                        rotation_angle = np.pi
                else:
                    # Normalize rotation axis
                    rotation_axis = rotation_axis / torch.norm(rotation_axis)
                    # Calculate rotation angle
                    rotation_angle = torch.acos(torch.clamp(torch.dot(default_axis, polar_axis), -1.0, 1.0))
                
                # Convert axis-angle to quaternion
                qx = rotation_axis[0] * torch.sin(rotation_angle / 2)
                qy = rotation_axis[1] * torch.sin(rotation_angle / 2)
                qz = rotation_axis[2] * torch.sin(rotation_angle / 2)
                qw = torch.cos(rotation_angle / 2)
                
                q = torch.tensor([qw, qx, qy, qz], device=device)
                
                # Apply rotation to all directions using the quaternion
                ray_directions = quat_apply(q.repeat(self.spherical2_num_points, 1), ray_directions)
            
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")

        return ray_origins, ray_directions


@dataclass
class RayCasterCfg:
    """Configuration for ray caster."""
    # Patterns and geometry
    pattern_cfg: RayCasterPatternCfg = field(default_factory=RayCasterPatternCfg)

    # Mesh to ray cast against - can be either file paths or vertices/triangles
    mesh_paths: List[str] = field(default_factory=list)
    vertices: torch.Tensor = None  # Vertices of the mesh as tensor of shape (N, 3)
    triangles: torch.Tensor = None  # Triangle indices of the mesh as tensor of shape (M, 3)

    # Ray casting properties
    max_distance: float = 100.0
    attach_yaw_only: bool = True  # If True, only yaw rotation is applied to rays

    # Sensor position offset
    offset_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    offset_rot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # quaternion

    # Sensor update settings
    update_period: float = 0.0  # 0.0 means update every step


@dataclass
class RayCasterData:
    """Data from ray caster."""
    ray_hits: torch.Tensor = None  # Ray hit positions (num_sensors, num_rays, 3)
    ray_hits_found: torch.Tensor = None  # Boolean mask for hits (num_sensors, num_rays)
    pos: torch.Tensor = None  # Sensor positions (num_sensors, 3)
    rot: torch.Tensor = None  # Sensor rotations as quaternions (num_sensors, 4)

    def __post_init__(self):
        """Initialize the data object with empty tensors."""
        pass  # Initialized in the ray caster


class RayCaster:
    """Ray casting sensor for Isaac Gym.

    This class implements a ray casting sensor that can be used with Isaac Gym environments.
    It uses NVIDIA WARP for efficient ray casting against meshes.
    """

    def __init__(self, cfg: RayCasterCfg, num_envs: int, device: str = "cuda:0"):
        """Initialize ray caster.

        Args:
            cfg: Ray caster configuration.
            num_envs: Number of environments.
            device: Device to allocate tensors on.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # For WARP, convert cuda:0 to cuda
        self.warp_device = "cpu" if device == "cpu" else "cuda"
        print(f"Initializing RayCaster with PyTorch device: {device}, WARP device: {self.warp_device}")

        self._is_initialized = False
        self._timestamp = torch.zeros(num_envs, device=device)
        self._timestamp_last_update = torch.zeros(num_envs, device=device)
        self._is_outdated = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Will be initialized later
        self.meshes = {}
        self.ray_origins = None
        self.ray_directions = None
        self.num_rays = 0
        self._data = RayCasterData()

        # Initialize the ray caster
        self._initialize()

    def _initialize(self):
        """Initialize the ray caster."""
        if self._is_initialized:
            return

        # Load meshes or create them from vertices/triangles
        self._initialize_meshes()

        # Initialize ray patterns
        self._initialize_rays()

        self._is_initialized = True

    def _initialize_meshes(self):
        """Load meshes for ray casting or create them from vertices/triangles."""
        if self.cfg.mesh_paths:
            for mesh_path in self.cfg.mesh_paths:
                try:
                    # Load mesh using trimesh
                    if os.path.isfile(mesh_path):
                        mesh = trimesh.load(mesh_path)
                    elif os.path.isfile(os.path.join(LEGGED_GYM_ROOT_DIR, mesh_path)):
                        mesh = trimesh.load(os.path.join(LEGGED_GYM_ROOT_DIR, mesh_path))
                    else:
                        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

                    # Convert to warp mesh - use the warp_device property
                    wp_mesh = convert_to_warp_mesh(
                        mesh.vertices,
                        mesh.faces,
                        device=self.warp_device
                    )

                    # Store the mesh
                    self.meshes[mesh_path] = wp_mesh

                    print(f"Loaded mesh: {mesh_path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                except Exception as e:
                    print(f"Failed to load mesh {mesh_path}: {e}")
        elif self.cfg.vertices is not None and self.cfg.triangles is not None:
            # Create a single warp mesh from vertices and triangles
            wp_mesh = convert_to_warp_mesh(
                self.cfg.vertices.cpu().numpy(),
                self.cfg.triangles.cpu().numpy(),
                device=self.warp_device
            )
            self.meshes["custom_mesh"] = wp_mesh
            print(f"Created custom mesh with {len(self.cfg.vertices)} vertices and {len(self.cfg.triangles)} triangles")
        else:
            raise ValueError("No mesh or vertices/triangles provided for ray casting.")

        if not self.meshes:
            raise RuntimeError("No meshes were successfully loaded or created.")

    def _initialize_rays(self):
        """Initialize ray patterns."""
        # Create ray patterns according to configuration
        ray_origins, ray_directions = self.cfg.pattern_cfg.create_pattern(self.device)
        self.num_rays = len(ray_directions)

        # Apply offset transformation to the rays
        offset_pos = torch.tensor(self.cfg.offset_pos, device=self.device)

        # Use the offset
        self.ray_origins = ray_origins + offset_pos
        self.ray_directions = ray_directions

        # Repeat for each environment
        self.ray_origins = self.ray_origins.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

        # Initialize data tensors
        self._data.pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._data.rot = torch.zeros(self.num_envs, 4, device=self.device)
        self._data.rot[:, 3] = 1.0  # Initialize to identity quaternion
        self._data.ray_hits = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self._data.ray_hits_found = torch.zeros(self.num_envs, self.num_rays, dtype=torch.bool, device=self.device)

    def update(self, dt: float, sensor_pos: torch.Tensor, sensor_rot: torch.Tensor, env_ids: torch.Tensor = None):
        """Update ray caster.

        Args:
            dt: Time step.
            sensor_pos: Sensor positions (num_envs, 3)
            sensor_rot: Sensor rotations as quaternions (num_envs, 4) in (w,x,y,z) format
            env_ids: Environment IDs to update. If None, all environments are updated.
        """
        if not self._is_initialized:
            self._initialize()

        # Update timestamp
        self._timestamp += dt

        # Determine which environments to update
        if env_ids is None:
            self._is_outdated |= (self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period)
            env_ids = self._is_outdated.nonzero().squeeze(-1)
        else:
            self._is_outdated[env_ids] = True

        if len(env_ids) == 0:
            return

        # Ensure consistent dtypes - convert input tensors to match data tensors
        sensor_pos_input = sensor_pos[env_ids].to(self._data.pos.dtype)
        sensor_rot_input = sensor_rot[env_ids].to(self._data.rot.dtype)

        # Update sensor data for the specified environments
        self._data.pos[env_ids] = sensor_pos_input
        self._data.rot[env_ids] = sensor_rot_input

        # Apply transformations and perform ray casting
        self._update_ray_casting(env_ids)

        # Update timestamps
        self._timestamp_last_update[env_ids] = self._timestamp[env_ids]
        self._is_outdated[env_ids] = False

    def _update_ray_casting(self, env_ids: torch.Tensor):
        """Update ray casting for the specified environments.

        Args:
            env_ids: Environment IDs to update.
        """

        # Get positions and rotations
        pos_w = self._data.pos[env_ids]
        quat_w = self._data.rot[env_ids]
        # Apply transformations based on sensor poses
        if self.cfg.attach_yaw_only:
            # Only yaw orientation is considered
            ray_origins_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_origins[env_ids])
            ray_origins_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            # FIXME
            # Full orientation is considered
            ray_origins_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_origins[env_ids])
            ray_origins_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        # Perform ray casting for each mesh
        # FIXME: For simplicity, we'll just use the first mesh in this implementation
        if self.meshes:
            mesh = next(iter(self.meshes.values()))
            hits, hits_found = raycast_mesh(
                ray_origins_w,
                ray_directions_w,
                max_dist=self.cfg.max_distance,
                mesh=mesh
            )

            # Store the results
            self._data.ray_hits[env_ids] = hits
            self._data.ray_hits_found[env_ids] = hits_found

    def reset(self, env_ids=None):
        """Reset ray caster for specified environments.

        Args:
            env_ids: Environment IDs to reset. If None, all environments are reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset timestamps
        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0
        self._is_outdated[env_ids] = True

    @property
    def data(self) -> RayCasterData:
        """Get ray caster data.

        Returns:
            Ray caster data.
        """
        return self._data


# Demo code to test the ray caster
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Test with mesh file
    def test_with_mesh_file():
        # Create a sample mesh (sphere)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh_path = "/tmp/sphere.obj"
        sphere.export(mesh_path)

        # Create ray caster configuration
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.GRID,
            grid_dims=(10, 10),
            grid_width=2.0,
            grid_height=2.0
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

        # Create sensor pose
        sensor_pos = torch.tensor([[0.0, 0.0, -3.0]], device=device)  # Below the sphere
        sensor_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # Identity rotation

        # Update ray caster
        ray_caster.update(0.1, sensor_pos, sensor_rot)

        # Get ray caster data
        data = ray_caster.data

        # Convert to numpy for visualization
        ray_hits = data.ray_hits[0].cpu().numpy()
        ray_hits_found = data.ray_hits_found[0].cpu().numpy()
        ray_origins = ray_caster.ray_origins[0].cpu().numpy()
        ray_directions = ray_caster.ray_directions[0].cpu().numpy()

        # Visualize
        visualize_rays(sphere.vertices, sphere.faces, ray_hits, ray_hits_found, 
                     sensor_pos[0].cpu().numpy(), ray_origins, ray_directions, ray_caster_cfg.max_distance)
    
    # Test with direct vertices and triangles
    def test_with_vertices_triangles():
        # Create a sample mesh (sphere)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        
        # Extract vertices and triangles
        vertices = torch.tensor(sphere.vertices, dtype=torch.float32)
        triangles = torch.tensor(sphere.faces, dtype=torch.int32)
        
        # Create ray caster configuration
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.GRID,
            grid_dims=(10, 10),
            grid_width=2.0,
            grid_height=2.0
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        vertices = vertices.to(device)
        triangles = triangles.to(device)
        
        ray_caster_cfg = RayCasterCfg(
            pattern_cfg=pattern_cfg,
            vertices=vertices,
            triangles=triangles,
            max_distance=10.0,
            offset_pos=[0.0, 0.0, 0.0]
        )

        # Create ray caster
        ray_caster = RayCaster(ray_caster_cfg, num_envs=1, device=device)

        # Create sensor pose
        sensor_pos = torch.tensor([[0.0, 0.0, -3.0]], device=device)  # Below the sphere
        sensor_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # Identity rotation

        # Update ray caster
        ray_caster.update(0.1, sensor_pos, sensor_rot)

        # Get ray caster data
        data = ray_caster.data

        # Convert to numpy for visualization
        ray_hits = data.ray_hits[0].cpu().numpy()
        ray_hits_found = data.ray_hits_found[0].cpu().numpy()
        ray_origins = ray_caster.ray_origins[0].cpu().numpy()
        ray_directions = ray_caster.ray_directions[0].cpu().numpy()

        # Visualize
        visualize_rays(sphere.vertices, sphere.faces, ray_hits, ray_hits_found, 
                     sensor_pos[0].cpu().numpy(), ray_origins, ray_directions, ray_caster_cfg.max_distance)
    
    def visualize_rays(vertices, faces, ray_hits, ray_hits_found, sensor_pos, ray_origins, ray_directions, max_distance):
        # Visualize
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original mesh
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            alpha=0.2
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

        # Plot ray origins and directions
        for i in range(len(ray_origins)):
            origin = sensor_pos + ray_origins[i]
            direction = ray_directions[i]

            # Draw ray
            if ray_hits_found[i]:
                end_point = ray_hits[i]
                ax.plot(
                    [origin[0], end_point[0]],
                    [origin[1], end_point[1]],
                    [origin[2], end_point[2]],
                    'g-',
                    alpha=0.3
                )
            else:
                end_point = origin + direction * max_distance
                ax.plot(
                    [origin[0], end_point[0]],
                    [origin[1], end_point[1]],
                    [origin[2], end_point[2]],
                    'b-',
                    alpha=0.1
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Ray Casting Demo')
        plt.tight_layout()
        plt.show()
    
    # Run tests
    print("Testing with mesh file...")
    test_with_mesh_file()
    
    print("Testing with direct vertices and triangles...")
    test_with_vertices_triangles()
