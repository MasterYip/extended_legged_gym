import numpy as np
import trimesh
import matplotlib.pyplot as plt


class RayTracer:
    def __init__(self, mesh):
        """Initialize ray tracer with a trimesh object"""
        self.mesh = mesh
        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    def trace_rays(self, origins, directions):
        """
        Trace rays against the mesh
        :param origins: (n,3) array of ray origins
        :param directions: (n,3) array of ray directions
        :return: hit_points (m,3) array of intersection points
        """
        locations, index_ray, index_tri = self.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions)
        return locations


# Demo: Ray tracing on a sphere mesh with visualization
if __name__ == "__main__":
    # Create sample mesh (sphere)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    # Initialize ray tracer
    tracer = RayTracer(mesh)

    # Generate grid of rays pointing towards the sphere
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx) * -3  # Rays start below the sphere

    ray_origins = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    ray_directions = np.array([[0, 0, 1]]*400)  # All rays point upwards
    print(ray_origins)
    print(ray_directions)

    # Perform ray tracing
    hits = tracer.trace_rays(ray_origins, ray_directions)
    print(hits)
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original mesh
    mesh.show(smooth=False, axes=ax)

    # Plot ray hits
    if len(hits) > 0:
        ax.scatter(hits[:, 0], hits[:, 1], hits[:, 2],
                   c='r', s=20, label='Intersection Points')

    ax.set_title('Ray Tracing Demo - Sphere Intersections')
    ax.legend()
    plt.show()
