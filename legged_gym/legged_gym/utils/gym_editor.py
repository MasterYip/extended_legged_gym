import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch
import random
import math


class ObstacleGenConfig:
    """Configuration class for ObstacleGen.
    
    This class defines parameters for passive stone obstacle generation in the simulation.
    """
    def __init__(self):
        # Stone types and their probabilities
        self.stone_types = ["box", "sphere", "capsule"]  # Available stone types
        self.type_probabilities = [0.6, 0.3, 0.1]  # Probability of each type being selected
        
        # Size ranges for different stone types (min, max)
        self.box_size_range = [0.08, 0.25]  # Size range for box stones
        self.sphere_radius_range = [0.05, 0.15]  # Radius range for sphere stones
        self.capsule_radius_range = [0.03, 0.08]  # Radius range for capsule stones
        self.capsule_length_range = [0.1, 0.2]  # Length range for capsule stones
        
        # Physical properties
        self.density_range = [800, 2000]  # Density range for stones (kg/m^3)
        self.restitution_range = [0.1, 0.4]  # Restitution (bounciness) range
        self.friction_range = [0.3, 0.9]  # Friction range
        
        # Spawn parameters
        self.spawn_height_range = [0.3, 1.0]  # Height range for spawning stones
        self.spawn_radius_range = [1.5, 6.0]  # Radial distance range from robot
        self.max_stones_per_env = 15  # Maximum number of stones per environment
        self.min_stones_per_env = 5  # Minimum number of stones per environment
        
        # Initial velocity parameters (for dropping effect)
        self.initial_horizontal_vel_range = [-0.5, 0.5]  # Initial horizontal velocity range (m/s)
        self.initial_vertical_vel_range = [-0.2, 0.0]  # Initial vertical velocity range (m/s)
        
        # Visual parameters - stone-like colors
        self.color_options = [
            gymapi.Vec3(0.6, 0.6, 0.6),  # Gray
            gymapi.Vec3(0.7, 0.7, 0.7),  # Light gray
            gymapi.Vec3(0.5, 0.5, 0.5),  # Dark gray
            gymapi.Vec3(0.6, 0.5, 0.4),  # Brown
            gymapi.Vec3(0.7, 0.6, 0.5),  # Light brown
            gymapi.Vec3(0.5, 0.4, 0.3),  # Dark brown
            gymapi.Vec3(0.4, 0.4, 0.4),  # Charcoal
        ]
        
        # Distribution parameters
        self.cluster_probability = 0.3  # Probability of spawning stones in clusters
        self.cluster_size_range = [2, 5]  # Number of stones in a cluster
        self.cluster_radius_range = [0.3, 1.0]  # Radius of the cluster


class ObstacleGen:
    """Class for generating passive stone obstacles in Isaac Gym environments.
    
    This class handles the creation of passive stone obstacles that can interact 
    with the robot and terrain through physics simulation.
    """
    def __init__(self, gym, sim, envs, cfg=None, device="cuda:0"):
        """Initialize the stone obstacle generator.
        
        Args:
            gym: The gymapi instance
            sim: The simulation instance
            envs: List of environment handles
            cfg: ObstacleGenConfig instance (optional, will use default if None)
            device: Device to use for torch tensors ("cuda:0" or "cpu")
        """
        self.gym = gym
        self.sim = sim
        self.envs = envs
        self.cfg = cfg if cfg is not None else ObstacleGenConfig()
        self.device = device
        self.device_type = device.split(':')[0]
        
        self.num_envs = len(envs)
        
        # Initialize stone tracking
        self.stones = [[] for _ in range(self.num_envs)]  # List of stones per environment
        
    def generate_stones(self, robot_positions=None):
        """Generate stone obstacles in all environments.
        
        Args:
            robot_positions: Tensor of robot base positions (num_envs, 3) for relative placement
        """
        for env_id in range(self.num_envs):
            # Determine number of stones to spawn
            num_stones = random.randint(self.cfg.min_stones_per_env, self.cfg.max_stones_per_env)
            
            if robot_positions is not None:
                robot_pos = robot_positions[env_id].cpu().numpy()
            else:
                robot_pos = None
                
            # Generate stones (potentially in clusters)
            stones_to_generate = num_stones
            while stones_to_generate > 0:
                # Decide if we should create a cluster
                if random.random() < self.cfg.cluster_probability and stones_to_generate > 1:
                    # Create a cluster of stones
                    cluster_size = min(random.randint(*self.cfg.cluster_size_range), stones_to_generate)
                    self._spawn_stone_cluster(env_id, cluster_size, robot_pos)
                    stones_to_generate -= cluster_size
                else:
                    # Create a single stone
                    self._spawn_stone(env_id, robot_pos)
                    stones_to_generate -= 1
    
    def reset(self, env_ids=None):
        """Reset stones for specified environments.
        
        Args:
            env_ids: List of environment IDs to reset (if None, reset all)
        """
        if env_ids is None:
            env_ids = range(self.num_envs)
        
        for env_id in env_ids:
            # Remove all stones in this environment
            for stone in self.stones[env_id]:
                self.gym.destroy_actor(self.envs[env_id], stone)
            
            # Clear stone lists
            self.stones[env_id] = []
            
            # Generate new stones
            num_stones = random.randint(self.cfg.min_stones_per_env, self.cfg.max_stones_per_env)
            for _ in range(num_stones):
                self._spawn_stone(env_id)
    
    def _spawn_stone_cluster(self, env_id, cluster_size, robot_pos=None):
        """Spawn a cluster of stones in the specified environment.
        
        Args:
            env_id: Environment ID to spawn the stones in
            cluster_size: Number of stones in the cluster
            robot_pos: Position of the robot (optional, for relative spawning)
        """
        # Determine cluster center
        if robot_pos is None:
            # If robot position is not provided, use a random position
            center_x = random.uniform(-5, 5)
            center_y = random.uniform(-5, 5)
        else:
            # Spawn relative to robot position
            spawn_radius = random.uniform(*self.cfg.spawn_radius_range)
            spawn_angle = random.uniform(0, 2 * math.pi)
            center_x = robot_pos[0] + spawn_radius * math.cos(spawn_angle)
            center_y = robot_pos[1] + spawn_radius * math.sin(spawn_angle)
        
        # Spawn stones in the cluster
        cluster_radius = random.uniform(*self.cfg.cluster_radius_range)
        for _ in range(cluster_size):
            # Random position within cluster radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, cluster_radius)
            pos_x = center_x + distance * math.cos(angle)
            pos_y = center_y + distance * math.sin(angle)
            
            # Slight height variation
            height_offset = random.uniform(-0.1, 0.1)
            
            self._spawn_stone(env_id, robot_pos, (pos_x, pos_y, height_offset))
    
    def _spawn_stone(self, env_id, robot_pos=None, specific_pos=None):
        """Spawn a new stone in the specified environment.
        
        Args:
            env_id: Environment ID to spawn the stone in
            robot_pos: Position of the robot (optional, for relative spawning)
            specific_pos: Specific position to spawn the stone (optional, for clusters)
        """
        # Choose stone type based on probabilities
        stone_type = random.choices(
            self.cfg.stone_types, 
            weights=self.cfg.type_probabilities, 
            k=1
        )[0]
        
        # Create asset options
        asset_options = gymapi.AssetOptions()
        asset_options.density = random.uniform(*self.cfg.density_range)
        # asset_options.restitution = random.uniform(*self.cfg.restitution_range)
        # asset_options.friction = random.uniform(*self.cfg.friction_range)
        asset_options.linear_damping = 0.05
        asset_options.angular_damping = 0.05
        
        # Create asset based on type
        if stone_type == "box":
            # For box stones, vary the dimensions to make them more rock-like
            sx = random.uniform(*self.cfg.box_size_range)
            sy = random.uniform(*self.cfg.box_size_range)
            sz = random.uniform(*self.cfg.box_size_range)
            stone_asset = self.gym.create_box(
                self.sim, 
                sx, 
                sy, 
                sz, 
                asset_options
            )
        elif stone_type == "sphere":
            radius = random.uniform(*self.cfg.sphere_radius_range)
            stone_asset = self.gym.create_sphere(
                self.sim, 
                radius, 
                asset_options
            )
        elif stone_type == "capsule":
            radius = random.uniform(*self.cfg.capsule_radius_range)
            length = random.uniform(*self.cfg.capsule_length_range)
            stone_asset = self.gym.create_capsule(
                self.sim, 
                radius, 
                length, 
                asset_options
            )
        
        # Determine spawn position
        if specific_pos is not None:
            spawn_x, spawn_y, height_offset = specific_pos
            spawn_z = random.uniform(*self.cfg.spawn_height_range) + height_offset
        elif robot_pos is None:
            # If robot position is not provided, use a random position
            spawn_x = random.uniform(-5, 5)
            spawn_y = random.uniform(-5, 5)
            spawn_z = random.uniform(*self.cfg.spawn_height_range)
        else:
            # Spawn relative to robot position
            spawn_radius = random.uniform(*self.cfg.spawn_radius_range)
            spawn_angle = random.uniform(0, 2 * math.pi)
            spawn_x = robot_pos[0] + spawn_radius * math.cos(spawn_angle)
            spawn_y = robot_pos[1] + spawn_radius * math.sin(spawn_angle)
            spawn_z = random.uniform(*self.cfg.spawn_height_range)
        
        # Create pose for the stone
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(spawn_x, spawn_y, spawn_z)
        
        # Random rotation (makes it more natural)
        roll = random.uniform(0, math.pi)
        pitch = random.uniform(0, math.pi)
        yaw = random.uniform(0, 2 * math.pi)
        
        quat = gymapi.Quat.from_euler_zyx(yaw, pitch, roll)
        pose.r = quat
        
        # Create stone actor
        stone_handle = self.gym.create_actor(
            self.envs[env_id], 
            stone_asset, 
            pose, 
            f"stone_{len(self.stones[env_id])}", 
            env_id, 
            0,  # Collision group
            0   # Collision filter
        )
        
        # Apply stone-like color
        color = random.choice(self.cfg.color_options)
        self.gym.set_rigid_body_color(
            self.envs[env_id], 
            stone_handle, 
            0, 
            gymapi.MESH_VISUAL_AND_COLLISION, 
            color
        )
        
        # Apply initial velocity (for a more natural dropping effect)
        vel_x = random.uniform(*self.cfg.initial_horizontal_vel_range)
        vel_y = random.uniform(*self.cfg.initial_horizontal_vel_range)
        vel_z = random.uniform(*self.cfg.initial_vertical_vel_range)
        
        # # Get rigid body state tensor
        # rb_state_tensor = self.gym.get_actor_rigid_body_states(
        #     self.envs[env_id], 
        #     stone_handle, 
        #     gymapi.STATE_ALL
        # )
        
        # # Update velocity in the tensor
        # rb_state_tensor['vel'][0][0] = vel_x
        # rb_state_tensor['vel'][0][1] = vel_y
        # rb_state_tensor['vel'][0][2] = vel_z
        
        # # Set the modified tensor back
        # self.gym.set_actor_rigid_body_states(
        #     self.envs[env_id], 
        #     stone_handle, 
        #     rb_state_tensor, 
        #     gymapi.STATE_ALL
        # )
        
        # Add to tracking list
        self.stones[env_id].append(stone_handle)
        
        return stone_handle