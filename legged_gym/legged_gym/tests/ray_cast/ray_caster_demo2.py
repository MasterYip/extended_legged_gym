#!/usr/bin/env python3

"""
Ray Caster Gym Example

This script demonstrates how to integrate the RayCaster with a Legged Gym environment.
"""

import os
import numpy as np
import isaacgym
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

from legged_gym.envs import *
from legged_gym.utils.ray_caster import (
    RayCaster,
    RayCasterCfg,
    RayCasterPatternCfg,
    PatternType
)

# This is a simplified version of how you would integrate the RayCaster
# with a Legged Gym environment.


class LeggedRobotWithRayCaster:
    """Example of how to integrate the RayCaster with a Legged Robot environment."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize the environment.

        In a real environment, you would initialize the simulation, the terrain,
        the robot, etc. Here we'll just initialize the ray caster.
        """
        # This would normally come from the initialization of the actual environment
        self.num_envs = 4  # Example number of environments
        self.device = sim_device
        self.dt = 1/60  # 60 Hz simulation

        # Create a sample mesh for ray casting
        self.mesh_path = self._create_test_mesh()

        # Initialize ray caster
        self.ray_caster = self._init_ray_caster()

        # In an actual environment, these would be set during initialization
        # and updated during stepping
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.base_quat[:, 3] = 1.0  # Identity quaternion

        # Example positions for visualization
        self.base_pos[0, 0] = 0.0  # x
        self.base_pos[0, 1] = 0.0  # y
        self.base_pos[0, 2] = 1.0  # z

        # Initialize ray caster
        self._update_ray_caster()

    def _create_test_mesh(self, output_dir="/tmp"):
        """Create a test mesh for ray casting.

        In a real environment, you would use the actual terrain mesh.

        Returns:
            Path to the mesh file.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create a simple mesh (a box with some height variation)
        vertices = np.array([
            [-10, -10, 0],
            [10, -10, 0],
            [10, 10, 0],
            [-10, 10, 0],
            [-5, -5, 2],
            [5, -5, 1],
            [5, 5, 2],
            [-5, 5, 1]
        ], dtype=np.float32)

        # Define the faces
        faces = np.array([
            # Bottom
            [0, 1, 2],
            [0, 2, 3],
            # Inner structure
            [4, 5, 6],
            [4, 6, 7],
        ], dtype=np.int32)

        # Create a trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save the mesh to a file
        mesh_path = os.path.join(output_dir, "test_terrain.obj")
        mesh.export(mesh_path)

        self.terrain_mesh = mesh
        return mesh_path

    def _init_ray_caster(self):
        """Initialize the ray caster.

        Returns:
            Ray caster instance.
        """
        # Create ray caster configuration
        pattern_cfg = RayCasterPatternCfg(
            pattern_type=PatternType.CONE,
            cone_num_rays=24,
            cone_angle=30.0
        )

        ray_caster_cfg = RayCasterCfg(
            pattern_cfg=pattern_cfg,
            mesh_paths=[self.mesh_path],
            max_distance=20.0,
            offset_pos=[0.3, 0.0, 0.5],  # Offset from robot base
            attach_yaw_only=True,  # Only yaw orientation affects the ray directions
            update_period=0.1  # Update every 100ms
        )

        # Create ray caster
        ray_caster = RayCaster(ray_caster_cfg, self.num_envs, self.device)
        return ray_caster

    def step(self, actions):
        """Step the environment.

        In a real environment, you would step the simulation, compute rewards,
        check for terminations, etc. Here we'll just update the ray caster.

        Args:
            actions: Robot actions.

        Returns:
            Tuple of observations, rewards, dones, infos.
        """
        # In an actual environment, these would be updated from the simulation
        # Here we'll just move the robot a bit
        self.base_pos[:, 0] += 0.01  # Move forward in x

        # Update ray caster
        self._update_ray_caster()

        # Get ray caster data for observations
        ray_data = self.ray_caster.data

        # Here you would normally compute observations, rewards, dones, infos
        # And include the ray caster data in the observations
        return None, None, None, None

    def _update_ray_caster(self):
        """Update the ray caster with the current robot pose."""
        # Update the ray caster with the current robot pose
        self.ray_caster.update(self.dt, self.base_pos, self.base_quat)

    def reset(self, env_ids=None):
        """Reset the environment.

        In a real environment, you would reset the simulation, robot, etc.
        Here we'll just reset the ray caster.

        Args:
            env_ids: Environment IDs to reset. If None, all environments are reset.
        """
        # Reset ray caster
        self.ray_caster.reset(env_ids)

        # Reset robot pose
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset base pose
        self.base_pos[env_ids] = 0.0
        self.base_pos[env_ids, 2] = 1.0  # Set height
        self.base_quat[env_ids] = 0.0
        self.base_quat[env_ids, 3] = 1.0  # Identity quaternion

        # Update ray caster with reset pose
        self._update_ray_caster()

    def get_observations(self):
        """Get observations from the environment.

        In a real environment, you would compute observations from the simulation.
        Here we'll just return the ray caster data.

        Returns:
            Dict of observations.
        """
        # Get ray caster data
        ray_data = self.ray_caster.data

        # Compute distances from ray hits
        ray_dists = torch.norm(ray_data.ray_hits - ray_data.pos.unsqueeze(1), dim=2)
        ray_dists = ray_dists * ray_data.ray_hits_found + ray_data.ray_hits_found.float() * 100.0

        # In a real environment, you would include the ray caster data in the observations
        obs = {
            "ray_dists": ray_dists,
            "ray_hits_found": ray_data.ray_hits_found
        }

        return obs

    def visualize_ray_caster(self, env_idx=0):
        """Visualize the ray caster results for a specific environment.

        Args:
            env_idx: Environment index to visualize.

        Returns:
            Figure object.
        """
        # Get ray caster data
        data = self.ray_caster.data

        # Convert to numpy for visualization
        ray_hits = data.ray_hits[env_idx].cpu().numpy()
        ray_hits_found = data.ray_hits_found[env_idx].cpu().numpy()
        sensor_pos = data.pos[env_idx].cpu().numpy()

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the mesh
        ax.plot_trisurf(
            self.terrain_mesh.vertices[:, 0],
            self.terrain_mesh.vertices[:, 1],
            self.terrain_mesh.vertices[:, 2],
            triangles=self.terrain_mesh.faces,
            alpha=0.2
        )

        # Plot the sensor position (robot base)
        ax.scatter(
            sensor_pos[0],
            sensor_pos[1],
            sensor_pos[2],
            c='b',
            s=50,
            label='Robot Base'
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

        # Draw rays
        for i in range(len(ray_hits)):
            end_point = ray_hits[i]
            line_style = 'g-' if ray_hits_found[i] else 'r--'
            alpha = 0.5 if ray_hits_found[i] else 0.2
            ax.plot(
                [sensor_pos[0], end_point[0]],
                [sensor_pos[1], end_point[1]],
                [sensor_pos[2], end_point[2]],
                line_style,
                alpha=alpha
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Ray Casting Visualization')

        # Set equal aspect ratio
        max_range = np.array([
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
            ax.get_zlim()[1] - ax.get_zlim()[0]
        ]).max() / 2.0

        mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
        mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
        mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.tight_layout()

        return fig

# Example usage of the RayCasterGymExample


def main():
    """Run the example."""
    # Create a mock environment with ray caster
    env = LeggedRobotWithRayCaster(
        cfg=None,
        sim_params=None,
        physics_engine=None,
        sim_device="cuda:0" if torch.cuda.is_available() else "cpu",
        headless=False
    )

    # Reset the environment
    env.reset()

    # Step the environment a few times
    for i in range(10):
        print(f"Step {i}")
        env.step(None)  # No actions in this example

    # Visualize the ray caster results
    fig = env.visualize_ray_caster()
    plt.show()


if __name__ == "__main__":
    main()
