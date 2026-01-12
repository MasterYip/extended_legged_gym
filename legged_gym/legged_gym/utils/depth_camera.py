import torch
import torchvision
import numpy as np
from isaacgym import gymapi, gymtorch
import cv2
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
import trimesh
import warp as wp
from legged_gym.utils.ray_caster import convert_to_warp_mesh

class DepthCameraBase(ABC):
    """
    Base class for depth camera implementations.
    Defines the common interface for all depth camera types.
    """
    
    def __init__(self, cfg, device, num_envs):
        """
        Initialize the depth camera system.
        
        Args:
            cfg: Configuration object with depth camera settings
            device: PyTorch device (cuda/cpu)
            num_envs: Number of environments
        """
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        
        # Initialize resize transform
        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg.resized[1], self.cfg.resized[0]), 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )
        
        # Initialize depth buffer
        self.depth_buffer = torch.zeros(
            self.num_envs,  
            self.cfg.buffer_len, 
            self.cfg.resized[1], 
            self.cfg.resized[0]
        ).to(self.device)
    
    @abstractmethod
    def create_camera(self, env_handle, actor_handle, env_id=None):
        """Create a camera for the specified environment and actor."""
        pass
        
    @abstractmethod
    def update_depth_buffer(self, envs, episode_length_buf):
        """Update depth buffer with new camera images."""
        pass
    
    def normalize_depth_image(self, depth_image):
        """
        Originally, depth images are in range [-far_clip, -near_clip].
        Normalize depth image to range [-0.5, 0.5] (near_clip to far_clip).
        
        Args:
            depth_image: Raw depth image tensor
            
        Returns:
            Normalized depth image tensor
        """
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.near_clip) / (self.cfg.far_clip - self.cfg.near_clip) - 0.5
        return depth_image
    
    def crop_depth_image(self, depth_image):
        """
        Crop depth image by removing edges.
        
        Args:
            depth_image: Input depth image tensor
            
        Returns:
            Cropped depth image tensor
        """
        # Crop 30 pixels from left/right and 20 pixels from bottom
        return depth_image[:-2, 4:-4]
    
    def process_depth_image(self, depth_image, env_id=None):
        """
        Process raw depth image: crop, add noise, resize, and normalize.
        Now supports both single images and batch processing.
        
        Args:
            depth_image: Raw depth image tensor
                        - Single: [height, width]
                        - Batch: [batch_size, height, width]
            env_id: Environment ID (can be None for batch processing)
            
        Returns:
            Processed depth image tensor (same batch structure as input)
        """
        is_batch = len(depth_image.shape) == 3
        
        if not is_batch:
            # Single image processing (original behavior)
            # Add noise if enabled
            if hasattr(self.cfg, 'dis_noise'):
                depth_image += self.cfg.dis_noise * 2 * (torch.rand(1) - 0.5)[0]
            
            # Clip depth values
            depth_image = torch.clip(depth_image, -self.cfg.far_clip, -self.cfg.near_clip)
            
            # Resize image
            if self.cfg.resized[0] != self.cfg.original[0] or self.cfg.resized[1] != self.cfg.original[1]:
                depth_image = self.resize_transform(depth_image[None, :]).squeeze()
            
            # Normalize
            depth_image = self.normalize_depth_image(depth_image)
            
            return depth_image
        else:
            # Batch processing
            batch_size = depth_image.shape[0]
            
            # Add noise if enabled - BATCH OPERATION
            if hasattr(self.cfg, 'dis_noise'):
                noise = self.cfg.dis_noise * 2 * (torch.rand(batch_size, device=depth_image.device) - 0.5)
                depth_image += noise.view(batch_size, 1, 1)
            
            # Clip depth values - BATCH OPERATION
            depth_image = torch.clip(depth_image, -self.cfg.far_clip, -self.cfg.near_clip)
            
            # Resize images - BATCH OPERATION
            if self.cfg.resized[0] != self.cfg.original[0] or self.cfg.resized[1] != self.cfg.original[1]:
                # Add channel dimension for resize transform
                depth_image = depth_image.unsqueeze(1)  # [batch_size, 1, height, width]
                depth_image = self.resize_transform(depth_image).squeeze(1)  # [batch_size, resized_height, resized_width]
            
            # Normalize - BATCH OPERATION
            depth_image = self.normalize_depth_image(depth_image)
            
            return depth_image
    
    def get_depth_observation(self):
        """
        Get current depth observation for RL training.
        
        Returns:
            Depth observation tensor
        """
        if self.cfg.camera_type is None:
            return None
        return self.depth_buffer[:, -2]  # Return second-to-last frame
    
    def visualize_depth(self, env_id=0, window_name="Depth Image"):
        """
        Visualize depth image.
        
        Args:
            env_id: Environment ID to visualize
            window_name: window name
        """
        if self.cfg.camera_type is None:
            print("Depth camera is not enabled!")
            return
        depth_image = self.depth_buffer[env_id, -1].cpu().numpy()
        try:
            if not hasattr(self, 'figure'):
                self.figure = plt.figure(figsize=(8, 6))
                plt.title(f'Depth Image - {window_name}')
            plt.imshow(depth_image, cmap='rainbow', vmin=-0.5, vmax=0.5)
            # if self.figure don't have colorbar, add it
            if not hasattr(self, 'colorbar'):
                self.colorbar = plt.colorbar(label='Normalized Depth')
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.25)
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
    
    def get_depth_buffer(self):
        """Get the full depth buffer."""
        return self.depth_buffer
    
    def is_enabled(self):
        """Check if depth camera is enabled."""
        return self.cfg.camera_type is not None


class DepthCameraFake(DepthCameraBase):
    """
    Fake depth camera implementation that returns infinity distance values.
    Useful for testing or when depth data is not needed but the interface is required.
    """
    
    def __init__(self, cfg, device, num_envs):
        """
        Initialize the fake depth camera system.
        
        Args:
            cfg: Configuration object with depth camera settings
            device: PyTorch device (cuda/cpu)
            num_envs: Number of environments
        """
        super().__init__(cfg, device, num_envs)
        
        # Initialize depth buffer with -0.5 values (representing zero-distance after normalization)
        self.depth_buffer = torch.full(
            (self.num_envs, self.cfg.buffer_len, self.cfg.resized[1], self.cfg.resized[0]),
            -0.5,  # -0.5 represents zero-distance in the normalized range [-0.5, 0.5]
            device=self.device
        )
        
        # Camera poses for each environment (not used, but kept for interface compatibility)
        self.camera_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.camera_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.camera_rot[:, 3] = 1.0  # Initialize to identity quaternion
        
        print("Created Fake depth camera that returns infinity distance values")
    
    def create_camera(self, env_handle, actor_handle, env_id=None):
        """
        Creates a fake camera (does nothing but maintains the interface).
        
        Args:
            env_handle: Environment handle
            actor_handle: Actor handle
            env_id: Environment ID
            
        Returns:
            None
        """
        return None
    
    def update_depth_buffer(self, envs, episode_length_buf):
        """
        No update needed for fake camera - keeps returning infinity values.
        
        Args:
            envs: List of environment handles
            episode_length_buf: Buffer tracking episode lengths
        """
        # Nothing to do - depth buffer is already filled with infinity values
        pass
    
    def update(self, dt, sensor_pos, sensor_rot, env_ids=None):
        """
        Update camera positions (no-op for fake camera).
        
        Args:
            dt: Time step
            sensor_pos: Sensor positions
            sensor_rot: Sensor rotations as quaternions
            env_ids: Environment IDs to update
        """
        # Nothing to do - just maintain interface compatibility
        pass


class DepthCameraWarp(DepthCameraBase):
    """
    Depth camera implementation using WARP ray casting.
    Uses efficient GPU-based ray casting for depth image generation.
    """
    
    def __init__(self, cfg, device, num_envs, terrain_vertices=None, terrain_triangles=None):
        """
        Initialize the WARP-based depth camera system.
        
        Args:
            cfg: Configuration object with depth camera settings
            device: PyTorch device (cuda/cpu)
            num_envs: Number of environments
            terrain_vertices: Terrain mesh vertices (optional)
            terrain_triangles: Terrain mesh triangles (optional)
        """
        super().__init__(cfg, device, num_envs)
        
        # Initialize WARP
        wp.init()
        wp.config.quiet = True
        
        # For WARP, we need to use "cuda" not "cuda:0"
        self.warp_device = "cpu" if device == "cpu" else "cuda"
        
        # Camera poses for each environment
        self.camera_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.camera_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.camera_rot[:, 3] = 1.0  # Initialize to identity quaternion
        
        # Actor handles for attaching cameras
        self.actor_handles = [None] * self.num_envs
        
        # Store terrain mesh data
        self.terrain_vertices = terrain_vertices
        self.terrain_triangles = terrain_triangles
        
        # Will be initialized later
        self.meshes = {}
        self.ray_origins = None
        self.ray_directions = None
        
        # Initialize the camera parameters
        self._initialize()
    
    def _initialize(self):
        """Initialize the camera system and load meshes."""
        # Create mesh from terrain if provided
        if self.terrain_vertices is not None and self.terrain_triangles is not None:
            self._create_terrain_mesh()
        
        # Generate ray grid based on camera parameters
        self._initialize_ray_grid()
    
    def _create_terrain_mesh(self):
        """Create WARP mesh from terrain vertices and triangles."""
        try:
            # Convert to warp mesh
            wp_mesh = convert_to_warp_mesh(
                self.terrain_vertices,
                self.terrain_triangles,
                device=self.warp_device
            )
            
            # Store the mesh
            self.meshes['terrain'] = wp_mesh
            
            print(f"Created WARP terrain mesh with {len(self.terrain_vertices)} vertices and {len(self.terrain_triangles)} triangles")
        except Exception as e:
            print(f"Failed to create terrain mesh: {e}")
    
    def _initialize_ray_grid(self):
        """Initialize ray patterns for camera grid."""
        if self.cfg.camera_type is None:
            return
            
        # Camera parameters
        width = self.cfg.original[0]
        height = self.cfg.original[1]
        hfov = self.cfg.horizontal_fov
        aspect_ratio = width / height
        
        # Calculate vertical FOV based on horizontal FOV and aspect ratio
        vfov = 2 * np.arctan(np.tan(np.radians(hfov) / 2) / aspect_ratio)
        vfov_degrees = np.degrees(vfov)
        
        # Create grid of rays
        rows, cols = height, width
        
        # Generate normalized pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, rows),
            torch.linspace(-1, 1, cols),
            indexing='ij'
        )
        
        # Adjust for FOV
        i = i * np.tan(np.radians(vfov_degrees/2))
        j = j * np.tan(np.radians(hfov/2))
        
        # Create direction vectors
        directions = torch.stack([
            torch.ones_like(i),  # x component (forward)
            j,                  # y component (left)
            i                  # z component (up)
        ], dim=-1)
        
        # Normalize directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # Create ray origins at (0,0,0)
        origins = torch.zeros_like(directions)
        
        # Store origins and directions
        self.ray_origins = origins.reshape(-1, 3).to(self.device)
        self.ray_directions = directions.reshape(-1, 3).to(self.device)
        
        # Repeat for each environment
        self.ray_origins = self.ray_origins.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)
        
        print(f"Initialized ray grid for DepthCameraWarp: {rows}x{cols} = {rows*cols} rays per environment")
    
    def create_camera(self, env_handle, actor_handle, env_id=None):
        """
        Create a virtual camera for an environment.
        For WARP-based camera, this just stores the actor handle for later use.
        
        Args:
            env_handle: Environment handle
            actor_handle: Actor handle to attach the camera to
            env_id: Environment ID
            
        Returns:
            None (no actual camera handle is created)
        """
        if self.cfg.camera_type is None:
            return None
            
        if env_id is not None and 0 <= env_id < self.num_envs:
            self.actor_handles[env_id] = actor_handle
        
        # Return None as we don't create actual camera handles
        return None
    
    def update_depth_buffer(self, envs, episode_length_buf):
        """
        Update depth buffer with new ray cast results.
        
        Args:
            envs: List of environment handles
            episode_length_buf: Buffer tracking episode lengths
        """
        if self.cfg.camera_type is None:
            return
        
        # Check if we have any meshes
        if not self.meshes:
            print("Warning: No meshes available for ray casting.")
            return
        
        mesh = self.meshes['terrain']
        
        # Flatten ray arrays for batch processing
        batch_size, n_rays, _ = self.ray_origins.shape
        
        # Perform ray casting against the mesh
        from isaacgym.torch_utils import quat_apply
        
        # Apply transformations based on camera poses - BATCH OPERATIONS
        # Expand rotations and positions for all rays
        camera_rot_expanded = self.camera_rot.unsqueeze(1).expand(-1, n_rays, -1)  # [num_envs, n_rays, 4]
        camera_pos_expanded = self.camera_pos.unsqueeze(1).expand(-1, n_rays, -1)  # [num_envs, n_rays, 3]
        
        # Apply rotation to ray origins and directions in batch
        env_rays_origin = quat_apply(
            camera_rot_expanded.reshape(-1, 4), 
            self.ray_origins.reshape(-1, 3)
        ).reshape(batch_size, n_rays, 3)
        
        env_rays_origin += camera_pos_expanded
        
        env_rays_direction = quat_apply(
            camera_rot_expanded.reshape(-1, 4), 
            self.ray_directions.reshape(-1, 3)
        ).reshape(batch_size, n_rays, 3)
        
        # Flatten for ray casting
        ray_origins_flat = env_rays_origin.reshape(-1, 3)
        ray_directions_flat = env_rays_direction.reshape(-1, 3)
        
        # Create inputs for WARP raycast function
        from legged_gym.utils.ray_caster import raycast_mesh
        
        # Cast rays against the mesh
        hits, hits_found = raycast_mesh(
            ray_origins=ray_origins_flat,
            ray_directions=ray_directions_flat,
            max_dist=self.cfg.far_clip,
            mesh=mesh
        )
        
        # Reshape results
        hits = hits.reshape(batch_size, n_rays, 3)
        hits_found = hits_found.reshape(batch_size, n_rays)
        
        # Calculate distances - BATCH OPERATIONS
        distances = torch.norm(hits - self.camera_pos.unsqueeze(1), dim=2)
        
        # Create depth images from distances - BATCH OPERATIONS
        depth_images = torch.ones(
            batch_size, 
            self.cfg.original[1],
            self.cfg.original[0]
        ).to(self.device) * -self.cfg.far_clip
        
        # Fill in depth values where hits were found - BATCH OPERATIONS
        depth_images_flat = depth_images.reshape(batch_size, -1)  # [batch_size, height*width]
        distances_masked = torch.where(hits_found, -distances, torch.tensor(-self.cfg.far_clip, device=self.device))
        depth_images_flat[:] = distances_masked
        depth_images = depth_images_flat.reshape(batch_size, self.cfg.original[1], self.cfg.original[0])
        
        # Process depth images - BATCH OPERATION
        processed_depths = self.process_depth_image(depth_images)  # Process all at once
        
        # Initialize or update depth buffer - BATCH OPERATIONS
        init_flags = episode_length_buf <= 1
        
        # For environments that need initialization
        if init_flags.any():
            init_envs = torch.where(init_flags)[0]
            for env_idx in init_envs:
                self.depth_buffer[env_idx] = torch.stack([processed_depths[env_idx]] * self.cfg.buffer_len, dim=0)
        
        # For environments that need updates
        update_flags = ~init_flags
        if update_flags.any():
            update_envs = torch.where(update_flags)[0]
            for env_idx in update_envs:
                self.depth_buffer[env_idx] = torch.cat([
                    self.depth_buffer[env_idx, 1:], 
                    processed_depths[env_idx].unsqueeze(0)
                ], dim=0)

    def update(self, dt, sensor_pos, sensor_rot, env_ids=None):
        """
        Update camera positions from the simulator.
        
        Args:
            dt: Time step
            sensor_pos: Sensor positions (robot base) (num_envs, 3)
            sensor_rot: Sensor rotations as quaternions (robot base) (num_envs, 4)
            env_ids: Environment IDs to update (optional)
        """
        if self.cfg.camera_type is None:
            return
            
        # Determine which environments to update
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        if len(env_ids) == 0:
            return
            
        # Get camera installation position from config
        from isaacgym.torch_utils import quat_apply, quat_mul
        
        # Camera installation offset
        if hasattr(self.cfg, 'position'):
            camera_offset = torch.tensor(self.cfg.position, device=self.device, dtype=self.camera_pos.dtype)
        else:
            camera_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=self.camera_pos.dtype)
            
        # Rotation offset - handle either 'rotation' (euler angles) or 'angle' (pitch range)
        if hasattr(self.cfg, 'rotation'):
            # Convert euler angles to quaternion if specified
            from scipy.spatial.transform import Rotation as R
            import numpy as np
            r = R.from_euler('xyz', self.cfg.rotation, degrees=True)
            quat_offset = torch.tensor(r.as_quat(), device=self.device, dtype=self.camera_rot.dtype)  # [x, y, z, w]
            # Convert to [w, x, y, z] format
            quat_offset = torch.tensor([quat_offset[3], quat_offset[0], quat_offset[1], quat_offset[2]], 
                                     device=self.device, dtype=self.camera_rot.dtype)
        elif hasattr(self.cfg, 'angle') and len(self.cfg.angle) == 2:
            # Use the middle of the angle range as the camera pitch
            import numpy as np
            from scipy.spatial.transform import Rotation as R
            pitch = - np.mean(self.cfg.angle)  # Take the average of the angle range
            r = R.from_euler('y', pitch, degrees=True)  # Pitch is around Y axis
            quat_offset = torch.tensor(r.as_quat(), device=self.device, dtype=self.camera_rot.dtype)  # [x, y, z, w]
            # Convert to [w, x, y, z] format
            quat_offset = torch.tensor([quat_offset[3], quat_offset[0], quat_offset[1], quat_offset[2]], 
                                     device=self.device, dtype=self.camera_rot.dtype)
        else:
            quat_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.camera_rot.dtype)  # Identity quaternion
            
        # BATCH OPERATIONS for all env_ids at once
        # Extract sensor data for selected environments and ensure correct dtype
        selected_sensor_pos = sensor_pos[env_ids].to(dtype=self.camera_pos.dtype)  # [len(env_ids), 3]
        selected_sensor_rot = sensor_rot[env_ids].to(dtype=self.camera_rot.dtype)  # [len(env_ids), 4]
        
        # Apply robot's rotation to camera offset to get position in world frame - BATCH
        world_offset = quat_apply(selected_sensor_rot, camera_offset.expand(len(env_ids), -1))
        
        # Combine rotations: base_rot * cam_offset_rot - BATCH
        combined_rot = quat_mul(selected_sensor_rot, quat_offset.expand(len(env_ids), -1))
        
        # Set final camera pose - BATCH (dtypes now match)
        self.camera_pos[env_ids] = selected_sensor_pos + world_offset
        self.camera_rot[env_ids] = combined_rot

    def get_camera_handles(self):
        """Get list of camera handles."""
        return getattr(self, 'cam_handles', [])


class DepthCamera(DepthCameraBase):
    """
    Decoupled depth camera class for Isaac Gym environments.
    Uses Isaac Gym's built-in camera sensors.
    """
    
    def __init__(self, cfg, gym, sim, device, num_envs):
        """
        Initialize the depth camera system.
        
        Args:
            cfg: Configuration object with depth camera settings
            gym: Isaac Gym instance
            sim: Simulation handle
            device: PyTorch device (cuda/cpu)
            num_envs: Number of environments
        """
        super().__init__(cfg, device, num_envs)
        self.gym = gym
        self.sim = sim
        
        # Initialize camera handles and buffers
        self.cam_handles = []
        self.cam_tensors = []
    
    def create_camera(self, env_handle, actor_handle, env_id=None):
        """
        Create and attach a depth camera to an actor in an environment.
        
        Args:
            env_handle: Environment handle from gym.create_env
            actor_handle: Actor handle from gym.create_actor
            env_id: Environment ID for position randomization
            
        Returns:
            camera_handle: Handle to the created camera sensor
        """
        if self.cfg.camera_type is None:
            return None
            
        # Set up camera properties
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.original[0]
        camera_props.height = self.cfg.original[1]
        camera_props.enable_tensors = True
        camera_props.horizontal_fov = self.cfg.horizontal_fov 

        # Create camera sensor
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        
        # Set up camera transform
        local_transform = gymapi.Transform()
        
        # Set camera position (with optional randomization)
        camera_position = np.copy(self.cfg.position)
        if hasattr(self.cfg, 'angle') and len(self.cfg.angle) == 2:
            camera_angle = np.random.uniform(self.cfg.angle[0], self.cfg.angle[1])
        else:
            camera_angle = 0
            
        local_transform.p = gymapi.Vec3(*camera_position)
        local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
        
        # Attach camera to robot body
        root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
        self.gym.attach_camera_to_body(
            camera_handle, 
            env_handle, 
            root_handle, 
            local_transform, 
            gymapi.FOLLOW_TRANSFORM
        )
        
        self.cam_handles.append(camera_handle)
        return camera_handle

    def update_depth_buffer(self, envs, episode_length_buf):
        """
        Update depth buffer with new camera images.
        
        Args:
            envs: List of environment handles
            episode_length_buf: Buffer tracking episode lengths
        """
        if self.cfg.camera_type is None:
            return
            
        self.gym.step_graphics(self.sim)  # Required for headless rendering
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # Collect all depth images first for potential batch processing
        raw_depth_images = []
        for i in range(self.num_envs):
            # Get depth image from GPU tensor
            depth_image_ = self.gym.get_camera_image_gpu_tensor(
                self.sim, 
                envs[i], 
                self.cam_handles[i],
                gymapi.IMAGE_DEPTH
            )
            depth_image = gymtorch.wrap_tensor(depth_image_)
            raw_depth_images.append(depth_image)

        # Stack for batch processing if all images have same shape
        if len(raw_depth_images) > 0:
            try:
                # Try batch processing
                stacked_images = torch.stack(raw_depth_images, dim=0)  # [num_envs, height, width]
                processed_depths = self.process_depth_image(stacked_images)  # Batch process
                
                # Update depth buffer - BATCH OPERATIONS where possible
                init_flags = episode_length_buf <= 1
                
                for i in range(self.num_envs):
                    if init_flags[i]:
                        self.depth_buffer[i] = torch.stack([processed_depths[i]] * self.cfg.buffer_len, dim=0)
                    else:
                        self.depth_buffer[i] = torch.cat([
                            self.depth_buffer[i, 1:], 
                            processed_depths[i].unsqueeze(0)
                        ], dim=0)
                        
            except Exception as e:
                # Fallback to individual processing if batch processing fails
                print(f"Batch processing failed, falling back to individual processing: {e}")
                for i in range(self.num_envs):
                    depth_image = self.process_depth_image(raw_depth_images[i], i)
                    
                    # Initialize or update depth buffer
                    init_flag = episode_length_buf <= 1
                    if init_flag[i]:
                        self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.buffer_len, dim=0)
                    else:
                        self.depth_buffer[i] = torch.cat([
                            self.depth_buffer[i, 1:], 
                            depth_image.to(self.device).unsqueeze(0)
                        ], dim=0)

        self.gym.end_access_image_tensors(self.sim)
    
    def get_depth_observation(self):
        """
        Get current depth observation for RL training.
        
        Returns:
            Depth observation tensor
        """
        if self.cfg.camera_type is None:
            return None
        return self.depth_buffer[:, -2]  # Return second-to-last frame
    
    def get_camera_handles(self):
        """Get list of camera handles."""
        return self.cam_handles

