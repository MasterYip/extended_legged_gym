from legged_gym.envs.base.legged_robot_raycast import LeggedRobotRayCast
from legged_gym.utils.depth_camera import DepthCamera, DepthCameraWarp, DepthCameraFake

class LeggedRobotDepth(LeggedRobotRayCast):
    """
    LeggedRobot with depth camera integration.
    Extends the base LeggedRobot class to support depth image acquisition with update decimation.
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Initialize the depth camera system
        self.depth_camera = None
        self.depth_update_counter = 0
        
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
            
    def _create_envs(self):
        """Override to add depth camera creation for each environment."""
        # Call parent method to create environments
        super()._create_envs()

        # Initialize depth camera based on camera_type
        if self.cfg.depth.camera_type is not None:
            if self.cfg.depth.camera_type == "IsaacGym":
                self.depth_camera = DepthCamera(
                    cfg=self.cfg.depth,
                    gym=self.gym,
                    sim=self.sim,
                    device=self.device,
                    num_envs=self.num_envs
                )
            elif self.cfg.depth.camera_type == "Warp":
                from legged_gym.utils.depth_camera import DepthCameraWarp
                
                # Get terrain vertices and triangles if available
                terrain_vertices = None
                terrain_triangles = None
                
                if hasattr(self, 'terrain'):
                    if self.cfg.terrain.mesh_type in ['trimesh', 'confined_trimesh']:
                        # Use the trimesh terrain
                        terrain_vertices = self.terrain.vertices.copy()
                        terrain_triangles = self.terrain.triangles.copy()
                        print(f"Using trimesh terrain mesh for depth camera: {len(terrain_vertices)} vertices, {len(terrain_triangles)} triangles")
                    elif self.cfg.terrain.mesh_type == 'heightfield':
                        # Convert heightfield to trimesh
                        from isaacgym import terrain_utils
                        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                            self.terrain.height_field_raw,
                            self.terrain.cfg.horizontal_scale,
                            self.terrain.cfg.vertical_scale,
                            self.cfg.terrain.slope_treshold
                        )
                        terrain_vertices = vertices
                        terrain_triangles = triangles
                        print(f"Converted heightfield to mesh for depth camera: {len(terrain_vertices)} vertices, {len(terrain_triangles)} triangles")

                    # Offset border_size to align the terrain
                    terrain_vertices[:, 0] -= self.cfg.terrain.border_size
                    terrain_vertices[:, 1] -= self.cfg.terrain.border_size
                    
                if terrain_vertices is None and self.cfg.terrain.mesh_type == 'plane':
                    # Create a simple ground plane mesh
                    import numpy as np
                    
                    # Define a large enough ground plane
                    size = 100.0
                    terrain_vertices = np.array([
                        [-size, -size, 0.0],
                        [size, -size, 0.0],
                        [size, size, 0.0],
                        [-size, size, 0.0]
                    ], dtype=np.float32)
                    
                    # Two triangles to form a rectangle
                    terrain_triangles = np.array([
                        [0, 1, 2],
                        [0, 2, 3]
                    ], dtype=np.int32)
                    
                    print(f"Created ground plane mesh for depth camera")
                
                self.depth_camera = DepthCameraWarp(
                    cfg=self.cfg.depth,
                    device=self.device,
                    num_envs=self.num_envs,
                    terrain_vertices=terrain_vertices,
                    terrain_triangles=terrain_triangles
                )
            elif self.cfg.depth.camera_type == "Fake":
                self.depth_camera = DepthCameraFake(
                    cfg=self.cfg.depth,
                    device=self.device,
                    num_envs=self.num_envs
                )
            else:
                print(f"Warning: Unknown camera type '{self.cfg.depth.camera_type}'. Depth camera disabled.")
                self.depth_camera = None

        # Create depth cameras for each environment if enabled
        if self.cfg.depth.camera_type is not None and self.depth_camera is not None:
            for i in range(self.num_envs):
                camera_handle = self.depth_camera.create_camera(
                    env_handle=self.envs[i],
                    actor_handle=self.actor_handles[i],
                    env_id=i
                )
    
    def post_physics_step(self):
        """Override to add depth camera update with decimation."""
        # Call parent method first
        super().post_physics_step()
        
        # Update depth camera with decimation
        if self.cfg.depth.camera_type is not None and self.depth_camera is not None:
            # Check if it's time to update depth images based on update_interval
            if self.depth_update_counter % self.cfg.depth.update_interval == 0:
                # For Warp camera, also update camera poses
                if self.cfg.depth.camera_type == "Warp":
                    self.depth_camera.update(
                        dt=self.dt,
                        sensor_pos=self.root_states[:, :3],
                        sensor_rot=self.root_states[:, 3:7]
                    )
                
                self.depth_camera.update_depth_buffer(self.envs, self.episode_length_buf)
            
            self.depth_update_counter += 1
    
    def compute_observations(self):
        """Override to optionally include depth observations."""
        # Call parent method to get base observations
        super().compute_observations()
        
        # Add depth observations if enabled (this is optional - depth can be used separately)
        # if self.cfg.depth.camera_type is not None and self.depth_camera is not None:
        #     depth_obs = self.depth_camera.get_depth_observation()
        #     if depth_obs is not None:
        #         # Flatten depth observation and concatenate with existing observations
        #         depth_obs_flat = depth_obs.view(self.num_envs, -1)
        #         self.obs_buf = torch.cat([self.obs_buf, depth_obs_flat], dim=1)
    
    def reset_idx(self, env_ids):
        """Override to handle depth buffer reset."""
        # Call parent reset method
        super().reset_idx(env_ids)
        
        # Reset depth camera buffers for reset environments if needed
        if self.cfg.depth.camera_type is not None and self.depth_camera is not None and len(env_ids) > 0:
            # The depth camera automatically handles initialization in update_depth_buffer
            # when episode_length_buf <= 1, so no explicit reset needed
            pass
    
    def get_depth_images(self):
        """
        Get current depth images from all environments.
        
        Returns:
            torch.Tensor or None: Depth buffer of shape (num_envs, buffer_len, height, width)
        """
        if self.cfg.depth.camera_type is not None and self.depth_camera is not None:
            return self.depth_camera.get_depth_buffer()
        return None
    
    def get_depth_observation(self):
        """
        Get depth observation for RL training.
        
        Returns:
            torch.Tensor or None: Current depth observation
        """
        if self.cfg.depth.camera_type is not None and self.depth_camera is not None:
            return self.depth_camera.get_depth_observation()
        return None
    
    def visualize_depth(self, env_id=0, window_name="Depth Image"):
        """
        Visualize depth image for debugging.
        
        Args:
            env_id (int): Environment ID to visualize
            window_name (str): OpenCV window name
        """
        if self.cfg.depth.camera_type is not None and self.depth_camera is not None:
            self.depth_camera.visualize_depth(env_id, window_name)
    
    def is_depth_enabled(self):
        """Check if depth camera is enabled and available."""
        return (self.cfg.depth.camera_type is not None and 
                self.depth_camera is not None and 
                self.depth_camera.is_enabled())
