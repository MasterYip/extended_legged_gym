import torch
import numpy as np
import trimesh
import warp as wp
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

# disable warp module initialization messages
wp.config.quiet = True
# initialize the warp module
wp.init()


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
def query_sdf_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    sdf_values: wp.array(dtype=wp.float32),
    sdf_gradients: wp.array(dtype=wp.vec3),
    max_distance: float,
    epsilon: float = 1.0e-3,
):
    """WARP kernel for querying SDF values and gradients from a mesh.

    Args:
        mesh: The mesh ID to query against.
        points: Array of query points.
        sdf_values: Output array to store SDF values.
        sdf_gradients: Output array to store SDF gradients.
        max_distance: Maximum distance to consider for SDF calculation.
        epsilon: Small value for numerical stability, as a fraction of the average edge length.
    """
    # Get thread ID
    tid = wp.tid()

    # Query the mesh for the closest point, which returns a mesh_query_point_t structure
    query_result = wp.mesh_query_point_sign_normal(
        mesh,            # The mesh identifier
        points[tid],     # The point in space to query
        max_distance,    # Maximum distance to consider
        epsilon          # Epsilon for numerical stability
    )

    # If a valid result was found
    if query_result.result:

        # Compute the closest point using mesh_eval_position with barycentric coordinates
        closest_point = wp.mesh_eval_position(
            mesh,
            query_result.face,
            query_result.u,
            query_result.v
        )

        # Calculate the distance vector
        dist_vec = points[tid] - closest_point

        # Calculate the distance (magnitude of the vector)
        distance = wp.length(dist_vec)

        # Apply the sign returned by the query
        signed_distance = distance * query_result.sign

        # Calculate normal using the normalized distance vector (for points outside)
        # or compute the face normal (for points inside)
        if distance > 1.0e-6:  # Point is not exactly on surface
            normal = dist_vec / distance
            # For inside points, flip the normal
            if query_result.sign < 0.0:
                normal = -normal
        else:
            # For points exactly on the surface, compute the face normal
            # Get the face vertices
            v0 = wp.mesh_eval_position(mesh, query_result.face, 0.0, 0.0)
            v1 = wp.mesh_eval_position(mesh, query_result.face, 1.0, 0.0)
            v2 = wp.mesh_eval_position(mesh, query_result.face, 0.0, 1.0)

            # Compute face normal
            face_normal = wp.cross(v1 - v0, v2 - v0)
            normal = wp.normalize(face_normal)

            # Ensure normal direction matches sign
            if query_result.sign < 0.0:
                normal = -normal

        # Store the SDF value and gradient
        sdf_values[tid] = signed_distance
        sdf_gradients[tid] = normal
    else:
        # If no hit, set to max distance and zero gradient
        sdf_values[tid] = max_distance
        sdf_gradients[tid] = wp.vec3(0.0, 0.0, 0.0)


@dataclass
class MeshSDFCfg:
    """Configuration for MeshSDF."""
    # Mesh to query against - can be either file paths or vertices/triangles
    mesh_paths: List[str] = field(default_factory=list)
    vertices: torch.Tensor = None  # Vertices of the mesh as tensor of shape (N, 3)
    triangles: torch.Tensor = None  # Triangle indices of the mesh as tensor of shape (M, 3)

    # Default SDF value for points far from the mesh
    default_sdf_value: float = 1000.0

    # Maximum distance to consider for SDF calculation
    max_distance: float = 100.0

    # Enable caching of query results (useful for static meshes)
    enable_caching: bool = False


@dataclass
class MeshSDFData:
    """Data from MeshSDF queries."""
    sdf_values: torch.Tensor = None  # SDF values (batch_size, num_points)
    sdf_gradients: torch.Tensor = None  # SDF gradients (batch_size, num_points, 3)

    def __post_init__(self):
        """Initialize the data object with empty tensors."""
        pass  # Initialized in the MeshSDF class


class MeshSDF:
    """Signed Distance Field (SDF) calculator using WARP for meshes.

    This class implements a signed distance field calculator that can efficiently
    compute SDF values and gradients for batches of points against a mesh.
    """

    def __init__(self, cfg: MeshSDFCfg, device: str = "cuda:0"):
        """Initialize MeshSDF.

        Args:
            cfg: MeshSDF configuration.
            device: Device to allocate tensors on.
        """
        self.cfg = cfg
        self.device = device

        # For WARP, convert cuda:0 to cuda
        self.warp_device = "cpu" if device == "cpu" else "cuda"
        print(f"Initializing MeshSDF with PyTorch device: {device}, WARP device: {self.warp_device}")

        self._is_initialized = False
        self._cache = {}  # For caching query results if enabled

        # Will be initialized later
        self.meshes = {}
        self._data = MeshSDFData()

        # Initialize the MeshSDF
        self._initialize()

    def _initialize(self):
        """Initialize the MeshSDF."""
        if self._is_initialized:
            return

        # Load meshes
        self._initialize_meshes()

        self._is_initialized = True

    def _initialize_meshes(self):
        """Load meshes for SDF calculation or create them from vertices/triangles."""
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
            raise ValueError("No mesh paths or vertices/triangles provided for SDF calculation.")

        if not self.meshes:
            raise RuntimeError("No meshes were successfully loaded or created.")

    def query(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query SDF values and gradients for points.

        Args:
            points: Tensor of shape (batch_size, num_points, 3) or (num_points, 3)

        Returns:
            Tuple of:
            - sdf_values: Tensor of shape (batch_size, num_points) or (num_points)
            - sdf_gradients: Tensor of shape (batch_size, num_points, 3) or (num_points, 3)
        """
        if not self._is_initialized:
            self._initialize()

        # Check input shapes and flatten if necessary
        input_rank = len(points.shape)
        if input_rank == 3:
            batch_size, num_points, _ = points.shape
            points_flat = points.reshape(-1, 3)
        elif input_rank == 2:
            batch_size, num_points = 1, points.shape[0]
            points_flat = points
        else:
            raise ValueError(f"Expected points to have rank 2 or 3, got {input_rank}")

        # Get device from input tensors
        device = points.device

        # Check if we have result in cache
        if self.cfg.enable_caching:
            cache_key = str(points_flat.cpu().numpy().data.tobytes())
            if cache_key in self._cache:
                sdf_values, sdf_gradients = self._cache[cache_key]
                return sdf_values, sdf_gradients

        # Prepare for WARP computation
        wp_device = self.warp_device
        total_points = points_flat.shape[0]
        points_np = points_flat.cpu().numpy()

        # Allocate output arrays
        points_wp = wp.array(points_np, dtype=wp.vec3, device=wp_device)
        sdf_values_wp = wp.zeros(total_points, dtype=wp.float32, device=wp_device)
        sdf_gradients_wp = wp.zeros(total_points, dtype=wp.vec3, device=wp_device)

        # Perform SDF query for each mesh
        # For simplicity, we'll just use the first mesh in this implementation
        if self.meshes:
            mesh = next(iter(self.meshes.values()))

            # Launch the kernel with correct parameters
            wp.launch(
                kernel=query_sdf_kernel,
                dim=total_points,
                inputs=[mesh.id, points_wp, sdf_values_wp, sdf_gradients_wp, float(self.cfg.max_distance), 1.0e-3],
                device=wp_device
            )

            # Synchronize to ensure computation is complete
            wp.synchronize()

            # Convert back to torch tensors
            sdf_values = torch.tensor(sdf_values_wp.numpy(), dtype=torch.float32, device=device)
            sdf_gradients = torch.tensor(sdf_gradients_wp.numpy(), dtype=torch.float32, device=device)

            # Reshape back if necessary
            if input_rank == 3:
                sdf_values = sdf_values.reshape(batch_size, num_points)
                sdf_gradients = sdf_gradients.reshape(batch_size, num_points, 3)

            # Cache the results if caching is enabled
            if self.cfg.enable_caching:
                self._cache[cache_key] = (sdf_values, sdf_gradients)

            return sdf_values, sdf_gradients
        else:
            # If no meshes are loaded, return default values
            if input_rank == 3:
                sdf_values = torch.ones(batch_size, num_points, device=device) * self.cfg.default_sdf_value
                sdf_gradients = torch.zeros(batch_size, num_points, 3, device=device)
            else:
                sdf_values = torch.ones(num_points, device=device) * self.cfg.default_sdf_value
                sdf_gradients = torch.zeros(num_points, 3, device=device)

            return sdf_values, sdf_gradients

    def nearest_points(self, query_points: torch.Tensor) -> torch.Tensor:
        """Find the nearest points on the mesh surface.

        Args:
            query_points: Tensor of shape (batch_size, num_points, 3) or (num_points, 3)

        Returns:
            nearest_points: Tensor of same shape as query_points with the nearest points on the surface
        """
        # Get SDF values and gradients
        sdf_values, sdf_gradients = self.query(query_points)

        # Calculate nearest points by moving along the gradient by the SDF distance
        if len(query_points.shape) == 3:
            # (batch_size, num_points, 3)
            nearest_points = query_points - sdf_values.unsqueeze(-1) * sdf_gradients
        else:
            # (num_points, 3)
            nearest_points = query_points - sdf_values.unsqueeze(-1) * sdf_gradients

        return nearest_points

    def clear_cache(self):
        """Clear the cache of query results."""
        self._cache.clear()

    @property
    def data(self) -> MeshSDFData:
        """Get MeshSDF data.

        Returns:
            MeshSDF data.
        """
        return self._data


# Demo code to test the MeshSDF
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a sample mesh (sphere)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_path = "/tmp/sphere.obj"
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

    # Create a grid of points to query
    x = torch.linspace(-2, 2, 20, device=device)
    y = torch.linspace(-2, 2, 20, device=device)
    z = torch.linspace(-2, 2, 20, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    # Query SDF values and gradients
    sdf_values, sdf_gradients = mesh_sdf.query(points)

    # Find points near the surface
    near_surface_mask = torch.abs(sdf_values) < 0.1
    near_surface_points = points[near_surface_mask]

    # Calculate nearest points on the surface
    nearest_points = mesh_sdf.nearest_points(points)
    nearest_surface_points = nearest_points[near_surface_mask]

    # Convert to numpy for visualization
    near_surface_points_np = near_surface_points.cpu().numpy()
    nearest_surface_points_np = nearest_surface_points.cpu().numpy()

    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original mesh
    ax.plot_trisurf(
        sphere.vertices[:, 0],
        sphere.vertices[:, 1],
        sphere.vertices[:, 2],
        triangles=sphere.faces,
        alpha=0.2
    )

    # Plot query points near the surface
    if len(near_surface_points_np) > 0:
        ax.scatter(
            near_surface_points_np[:, 0],
            near_surface_points_np[:, 1],
            near_surface_points_np[:, 2],
            c='r',
            s=20,
            label='Near Surface Points'
        )

    # Plot nearest points on the surface
    if len(nearest_surface_points_np) > 0:
        ax.scatter(
            nearest_surface_points_np[:, 0],
            nearest_surface_points_np[:, 1],
            nearest_surface_points_np[:, 2],
            c='g',
            s=20,
            label='Nearest Surface Points'
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('MeshSDF Demo')
    plt.legend()
    plt.tight_layout()
    plt.show()
