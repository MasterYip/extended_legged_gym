# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import torch
import numpy as np
import trimesh

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.batch_rollout.robot_batch_rollout import RobotBatchRollout
from legged_gym.envs.batch_rollout.robot_batch_rollout_percept_config import RobotBatchRolloutPerceptCfg, RobotBatchRolloutPerceptCfgPPO
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.ray_caster import RayCasterPatternCfg, RayCasterCfg, RayCaster, PatternType
from legged_gym.utils.mesh_sdf import MeshSDF, MeshSDFCfg


class RobotBatchRolloutPercept(RobotBatchRollout):
    """Batch rollout environment with perceptual features (raycast and SDF).

    This class extends RobotBatchRollout to add perception capabilities:
    1. Ray casting for distance sensing (e.g., lidar-like observations)
    2. Signed Distance Field (SDF) for proximity to mesh objects
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize environment with perceptual features.

        Args:
            cfg: Configuration object for the environment
            sim_params: Simulation parameters
            physics_engine: Physics engine to use
            sim_device: Device for simulation
            headless: Whether to run without rendering
        """
        # Initialize base class first (this will call create_sim)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         sim_device=sim_device,
                         headless=headless)

        # Initialize ray caster if enabled
        if hasattr(self.cfg, 'raycaster') and getattr(self.cfg.raycaster, "enable_raycast", False):
            self._init_ray_caster()

        # Initialize SDF if enabled
        if hasattr(self.cfg, 'sdf') and getattr(self.cfg.sdf, "enable_sdf", False):
            self._init_sdf()

    # def create_sim(self):
    #     """Creates simulation, terrain and environments, then initializes perception.
        
    #     Overrides parent class create_sim to add initialization of perceptual features
    #     after terrain is created.
    #     """
    #     # First create simulation, terrain and envs using parent method
    #     super().create_sim()
        
    #     # Now that terrain is created, initialize perceptual components
        
    #     # Initialize ray caster if enabled
    #     if hasattr(self.cfg, 'raycaster') and getattr(self.cfg.raycaster, "enable_raycast", False):
    #         self._init_ray_caster()

    #     # Initialize SDF if enabled
    #     if hasattr(self.cfg, 'sdf') and getattr(self.cfg.sdf, "enable_sdf", False):
    #         self._init_sdf()

    def _init_ray_caster(self):
        """Initialize ray caster with configuration from environment settings."""
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
        self.ray_caster = RayCaster(ray_caster_cfg, self.total_num_envs, self.device)

        # Calculate the observation size including raycast data
        self.num_ray_observations = self.ray_caster.num_rays

        # Initialize raycast distances tensor
        self.raycast_distances = torch.zeros(self.total_num_envs, self.num_ray_observations, device=self.device)

        print(f"Ray caster initialized with {pattern_type} pattern, {self.num_ray_observations} rays")

    def _init_sdf(self):
        """Initialize SDF with configuration from environment settings."""
        # Create SDF configuration
        sdf_cfg = MeshSDFCfg(
            max_distance=self.cfg.sdf.max_distance,
            enable_caching=self.cfg.sdf.enable_caching
        )
        
        # Use explicitly specified SDF mesh paths if available
        if hasattr(self.cfg.sdf, "mesh_paths") and self.cfg.sdf.mesh_paths:
            sdf_cfg.mesh_paths = self.cfg.sdf.mesh_paths
            print(f"Using SDF mesh paths: {self.cfg.sdf.mesh_paths}")
        # Fall back to terrain mesh file if terrain is using mesh files
        elif hasattr(self.cfg.terrain, "mesh_file") and self.cfg.terrain.mesh_file and hasattr(self.cfg.terrain, "use_terrain_obj") and self.cfg.terrain.use_terrain_obj:
            sdf_cfg.mesh_paths = [self.cfg.terrain.mesh_file]
            print(f"Using terrain mesh file for SDF: {self.cfg.terrain.mesh_file}")
        # Use procedurally generated terrain vertices and triangles
        elif hasattr(self, 'terrain') and hasattr(self.terrain, 'vertices') and hasattr(self.terrain, 'triangles'):
            # Convert terrain vertices and triangles to torch tensors if they're not already
            if isinstance(self.terrain.vertices, torch.Tensor):
                sdf_cfg.vertices = self.terrain.vertices
            else:
                sdf_cfg.vertices = torch.tensor(self.terrain.vertices, device=self.device)
            
            if isinstance(self.terrain.triangles, torch.Tensor):
                sdf_cfg.triangles = self.terrain.triangles
            else:
                sdf_cfg.triangles = torch.tensor(self.terrain.triangles, device=self.device)
            # Add border_size offset to align with terrain
            if hasattr(self.cfg.terrain, 'border_size'):
                sdf_cfg.vertices[:, 0] -= self.cfg.terrain.border_size
                sdf_cfg.vertices[:, 1] -= self.cfg.terrain.border_size
            print(f"Using procedurally generated terrain for SDF with {len(sdf_cfg.vertices)} vertices and {len(sdf_cfg.triangles)} triangles")
        elif self.cfg.terrain.mesh_type == 'plane':
            import numpy as np
            # Define a large enough ground plane
            size = 100.0
            sdf_cfg.vertices = np.array([
                [-size, -size, 0.0],
                [size, -size, 0.0],
                [size, size, 0.0],
                [-size, size, 0.0]
            ], dtype=np.float32)
            sdf_cfg.vertices = torch.tensor(sdf_cfg.vertices, device=self.device)
            
            # Two triangles to form a rectangle
            sdf_cfg.triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ], dtype=np.int32)
            sdf_cfg.triangles = torch.tensor(sdf_cfg.triangles, device=self.device)
            print("Using plane terrain for SDF")
        else:
            raise ValueError("No terrain mesh available for SDF calculation. Either set use_terrain_obj=True and provide a terrain file, or ensure terrain.vertices and terrain.triangles are available.")

        # Create SDF calculator
        self.mesh_sdf = MeshSDF(sdf_cfg, device=self.device)

        # Get the body indices for which to compute SDF
        self.sdf_body_indices = []
        for body_name in self.cfg.sdf.query_bodies:
            body_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            if body_idx >= 0:
                self.sdf_body_indices.append(body_idx)
            else:
                print(f"Warning: Body '{body_name}' not found for SDF query")

        # If no specific bodies were requested or found, use the base (root) body
        if not self.sdf_body_indices:
            self.sdf_body_indices = [0]  # Default to the root body
            print("No valid SDF query bodies found, defaulting to root body")

        # Initialize SDF value storage
        self.num_sdf_bodies = len(self.sdf_body_indices)
        self.sdf_values = torch.zeros(self.total_num_envs, self.num_sdf_bodies, device=self.device)
        self.sdf_gradients = torch.zeros(self.total_num_envs, self.num_sdf_bodies, 3, device=self.device)
        self.sdf_nearest_points = torch.zeros(self.total_num_envs, self.num_sdf_bodies, 3, device=self.device)

        # SDF update counter
        self.sdf_update_counter = 0

        print(f"SDF initialized with {self.num_sdf_bodies} query bodies")

    def _post_physics_step_callback(self):
        """Callback after physics step, updates perceptual features."""
        # Call parent class callback
        super()._post_physics_step_callback()

        # Update ray caster if enabled and available
        if (hasattr(self, 'ray_caster') and self.ray_caster is not None and 
            hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast):
            # Update ray caster with the current robot positions and orientations
            self.ray_caster.update(
                dt=self.dt,
                sensor_pos=self.base_pos,
                sensor_rot=self.base_quat,
            )
            # Get raycast points as observations
            self.raycast_distances = self._get_raycast_distances()

        # Update SDF values if enabled and available, based on update frequency
        if (hasattr(self, 'mesh_sdf') and self.mesh_sdf is not None and 
            hasattr(self.cfg.sdf, "enable_sdf") and self.cfg.sdf.enable_sdf):
            self.sdf_update_counter += 1
            if self.sdf_update_counter >= self.cfg.sdf.update_freq:
                self.sdf_update_counter = 0
                self._update_sdf_values()

    def _post_physics_step_callback_rollout(self):
        super()._post_physics_step_callback_rollout()

        # Update ray caster if enabled and available
        if (hasattr(self, 'ray_caster') and self.ray_caster is not None and 
            hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast):
            # Update ray caster with the current robot positions and orientations
            self.ray_caster.update(
                dt=self.dt,
                sensor_pos=self.base_pos,
                sensor_rot=self.base_quat,
            )
            # Get raycast points as observations
            self.raycast_distances = self._get_raycast_distances()

        # Update SDF values if enabled and available, based on update frequency
        if (hasattr(self, 'mesh_sdf') and self.mesh_sdf is not None and 
            hasattr(self.cfg.sdf, "enable_sdf") and self.cfg.sdf.enable_sdf):
            self.sdf_update_counter += 1
            if self.sdf_update_counter >= self.cfg.sdf.update_freq:
                self.sdf_update_counter = 0
                self._update_sdf_values()
        return


    def _get_raycast_distances(self, env_ids=None):
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

        # Normalize distances by max distance and invert (closer = higher value)
        max_distance = self.ray_caster.cfg.max_distance
        normalized_distances = 1.0 - torch.clamp(distances / max_distance, 0.0, 1.0)

        # For rays that didn't hit anything, set to 0 (maximum distance)
        normalized_distances = normalized_distances * hits_found.float()

        # Flatten to 1D per environment
        return normalized_distances.reshape(normalized_distances.shape[0], -1)

    def _update_sdf_values(self, env_ids=None):
        """Update SDF values for specified bodies.

        Args:
            env_ids: Optional indices of environments to update. Default is all environments.
        """
        # Use either specified env_ids or all environments
        if env_ids is None:
            env_ids = torch.arange(self.total_num_envs, device=self.device)

        # Get positions of the bodies to query
        body_positions = []
        for i, body_idx in enumerate(self.sdf_body_indices):
            # Get global position of the body from rigid body state tensor
            # Shape of rigid_body_state: [num_envs, num_bodies, 13] where 13 is [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            body_pos = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[env_ids, body_idx, 0:3]
            body_quat = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[env_ids, body_idx, 3:7]
            
            # Apply collision sphere position offset if configured
            if (hasattr(self.cfg.sdf, 'collision_sphere_pos') and 
                len(self.cfg.sdf.collision_sphere_pos) > i):
                # Get collision sphere offset for this body
                sphere_offset = torch.tensor(self.cfg.sdf.collision_sphere_pos[i], 
                                           device=self.device, dtype=torch.float32)
                
                # Transform offset from body frame to world frame using body quaternion
                sphere_offset_world = quat_rotate(body_quat, sphere_offset.unsqueeze(0).repeat(len(env_ids), 1))
                
                # Add offset to body position
                positions = body_pos + sphere_offset_world
            else:
                # Use body position directly if no offset configured
                positions = body_pos
            
            body_positions.append(positions)

        # Stack body positions to shape [num_env_ids, num_bodies, 3]
        query_points = torch.stack(body_positions, dim=1)

        # For each body, perform SDF query
        for i, body_idx in enumerate(self.sdf_body_indices):
            # Extract points for this body across all requested environments
            # Shape: [num_env_ids, 3]
            points = query_points[:, i, :]

            # Query SDF values, gradients, and nearest points
            # sdf_vals, sdf_grads = self.mesh_sdf.query(points, with_gradients=self.cfg.sdf.compute_gradients)
            sdf_vals, sdf_grads = self.mesh_sdf.query(points)

            if self.cfg.sdf.compute_nearest_points:
                nearest_pts = self.mesh_sdf.nearest_points(points)
                self.sdf_nearest_points[env_ids, i] = nearest_pts

            # Store the results
            self.sdf_values[env_ids, i] = sdf_vals
            if self.cfg.sdf.compute_gradients:
                self.sdf_gradients[env_ids, i] = sdf_grads

    def compute_observations(self):
        """Computes observations including perceptual data (raycasts and SDFs)."""
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

        # Add raycast data if enabled and available
        if (hasattr(self, 'ray_caster') and self.ray_caster is not None and 
            hasattr(self.cfg.raycaster, "enable_raycast") and self.cfg.raycaster.enable_raycast and
            hasattr(self, 'raycast_distances') and self.raycast_distances is not None):
            # Add normalized raycast distances to the observations
            self.obs_buf = torch.cat((self.obs_buf, self.raycast_distances), dim=-1)

        # Add SDF values if enabled and configured to be included in observations
        if (hasattr(self, 'mesh_sdf') and self.mesh_sdf is not None and 
            hasattr(self.cfg.sdf, "enable_sdf") and self.cfg.sdf.enable_sdf and 
            self.cfg.sdf.include_in_obs and hasattr(self, 'sdf_values') and self.sdf_values is not None):
            # Add SDF values (flattened) to the observations
            self.obs_buf = torch.cat((self.obs_buf, self.sdf_values), dim=-1)

        # Add noise if configured
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _draw_debug_vis(self):
        """Draw debug visualization for perceptual features."""
        # Call parent class visualization
        super()._draw_debug_vis()

        # Only draw if viewer and debug_viz are enabled
        if not (self.viewer and self.debug_viz):
            return

        # Draw ray visualization if enabled
        if hasattr(self, 'ray_caster'):
            self._draw_raycast_debug()

        # Draw SDF visualization if enabled
        if hasattr(self, 'mesh_sdf'):
            self._draw_sdf_debug()

    def _draw_raycast_debug(self):
        """Draw debug visualization for raycasts."""
        ray_dir_length = 2
        for k in range(self.num_envs):
            i = self.main_env_indices[k]
            base_pos = self.root_states[i, :3].cpu().numpy()
            # self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            # self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))

            # Draw ray visualization if enabled
            ray_dir_length = 2
            if hasattr(self, 'ray_caster'):
                ray_data = self.ray_caster.data
                # Get ray origins, directions and hits for this robot
                ray_dir = self.ray_caster.ray_directions[i, :]
                quat = self.root_states[i, 3:7].repeat(self.ray_caster.num_rays, 1)
                ray_origins = base_pos + quat_rotate(quat[:1, :], torch.tensor(self.cfg.raycaster.offset_pos,
                                                                               device=self.device).unsqueeze(0)).cpu().numpy()
                world_dir = quat_rotate(quat, ray_dir).cpu().numpy()
                for j in range(self.ray_caster.num_rays):
                    # Draw ray direction (even if no hit)
                    ray_dir_end = ray_origins + world_dir[j] * ray_dir_length
                    self.vis.draw_line(i, [ray_origins, ray_dir_end], color=(0.7, 0.7, 0.7))  # Gray for direction

        ray_hits = ray_data.ray_hits.view(-1, 3).cpu().numpy()
        ray_hits = ray_hits[ray_data.ray_hits_found.view(-1).cpu().numpy() > 0]  # Filter out hits that are not found
        self.vis.draw_points(i, ray_hits, color=(1, 0, 0), size=0.02)

    def _draw_sdf_debug(self):
        """Draw debug visualization for SDF values."""
        if not hasattr(self, 'sdf_values'):
            print("SDF values not available for visualization.")
            return

        for k in range(self.num_envs):
            i = self.main_env_indices[k]
            for j, body_idx in enumerate(self.sdf_body_indices):
                # Get body position
                body_pos = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[i, body_idx, 0:3].cpu().numpy()

                # Draw point at body position
                sdf_value = self.sdf_values[i, j].item()
                sdf_value*=2  # Scale SDF value for better visualization (optional, adjust as needed)
                # Create a continuous color map for SDF values)
                sdf_clipped = np.clip(sdf_value, -1.0, 1.0)
                if sdf_clipped < 0:
                    # Inside the mesh (red)
                    color = (1.0, 0.0, 0.0)
                else:
                    # Outside the mesh (green to cyan gradient)
                    color = (0.0, sdf_clipped, 1.0-sdf_clipped)  # Blue to green gradient

                # Draw a point showing the body position with color based on SDF value
                self.vis.draw_point(i, body_pos, color=color, size=0.05)

                # Draw gradient arrow if available
                if hasattr(self, 'sdf_gradients') and self.cfg.sdf.compute_gradients:
                    gradient = self.sdf_gradients[i, j].cpu().numpy()
                    gradient_length = 0.15  # Fixed length for visibility
                    gradient_magnitude = np.linalg.norm(gradient)

                    if gradient_magnitude > 1e-6:  # Only draw gradients with meaningful direction
                        # Draw the gradient direction
                        gradient_normalized = gradient / gradient_magnitude
                        arrow_end = body_pos + gradient_normalized * gradient_length
                        self.vis.draw_arrow(i, body_pos, arrow_end, width=0.003, color=color)

                # Draw line to nearest point if available
                if hasattr(self, 'sdf_nearest_points') and self.cfg.sdf.compute_nearest_points:
                    nearest_point = self.sdf_nearest_points[i, j].cpu().numpy()
                    self.vis.draw_line(i, [body_pos, nearest_point], color=(0.5, 0.5, 0.5))  # Gray line
                    self.vis.draw_point(i, nearest_point, color=(0.5, 0.6, 1.0), size=0.03)  # Blue point
