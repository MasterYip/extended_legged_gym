import torch
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *
from legged_gym.envs.elspider_air.pose_adapt.el_mini_base_pose_adapt import ElMiniBasePoseAdapt, ElMiniBasePoseAdaptCfg


class ElMiniBasePoseCtrlCfg(ElMiniBasePoseAdaptCfg):
    # Control method selection
    use_direct_pose_control: bool = True  # Using direct pose setting instead of PD control

    # Control frequency for sending commands
    control_decimation: int = 2  # Default control decimation

    class env:
        episode_length_s: float = 20  # Episode length in seconds
        num_envs: int = 1024
        num_observations: int = 132  # Will be updated dynamically based on actual obs
        num_privileged_obs: int = None
        num_actions: int = 6  # linear vel (3) + angular vel (3)
        env_spacing = 1

    class raycaster(ElMiniBasePoseAdaptCfg.raycaster):
        # Ray parameters
        draw_rays: bool = True
        draw_hits: bool = True
        draw_mesh: bool = True
        # Detailed ray pattern
        ray_pattern: str = "spherical"  # Can be "spherical", "radial", or "grid"
        spherical_num_azimuth: int = 32  # Double the number of rays for better terrain perception
        spherical_num_elevation: int = 16

    class terrain:
        terrain_name: str = "uneven_slope"
        max_init_terrain_level: int = 0
        terrain_length: float = 8.0
        terrain_width: float = 8.0
        num_rows: int = 20
        num_cols: int = 20
        # Randomization
        border_size: float = 0.0
        center_robots: bool = True
        center_span: float = 4.0
        # Steps
        horizontal_scale: float = 0.1
        vertical_scale: float = 0.005
        # Pyramid slope
        max_slope: float = 0.4
        platform_height: float = 0.1
        # Discrete
        discrete_obstacles_height: float = 0.05
        discrete_obstacles_width: float = 1.0


class ElMiniBasePoseCtrl(ElMiniBasePoseAdapt):
    """Implementation of the ElSpider Air Base Pose Controller.

    This class provides direct control of the base pose for the ElSpider Air robot,
    primarily for testing and visualization purposes.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Parent class initialization
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Set specific controller parameters
        self.use_direct_pose_control = cfg.use_direct_pose_control
        self.control_decimation = cfg.control_decimation
        self.control_counter = 0
        self._mesh_drawn = False

        # Initialize velocities
        self.lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize target positions and orientations for the controller
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 3] = 1.0  # Initialize as identity quaternion (w=1.0)

        # Initialize force and torque vectors
        self.external_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.external_torque = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize debug visualization attribute
        self.debug_viz = False

    def reset_idx(self, env_ids):
        """Reset environments with given IDs."""
        # Reset the base environment
        super().reset_idx(env_ids)

        # Reset controller-specific state variables
        self.lin_vel[env_ids] = 0
        self.ang_vel[env_ids] = 0
        self.target_pos[env_ids] = self.base_pos[env_ids].clone()
        self.target_quat[env_ids] = self.base_quat[env_ids].clone()

    def pre_physics_step(self, actions):
        """Prepare for physics step by applying actions."""
        # Convert actions to control commands
        lin_vel_commands = actions[:, :3]
        ang_vel_commands = actions[:, 3:6]

        # Update target position and orientation based on linear and angular velocities
        self.lin_vel = lin_vel_commands
        self.ang_vel = ang_vel_commands

        # Create rotation from angular velocity
        angular_velocity_norm = torch.norm(self.ang_vel, dim=1, keepdim=True)
        angular_velocity_norm = torch.clamp(angular_velocity_norm, min=1e-8)
        
        # Generate rotation axis
        rotation_axis = self.ang_vel / angular_velocity_norm
        
        # Calculate angular displacement for the timestep
        delta_angle = angular_velocity_norm * self.dt
        
        # Use axis-angle to quaternion conversion
        cos_angle = torch.cos(delta_angle * 0.5)
        sin_angle = torch.sin(delta_angle * 0.5)
        
        # Create the local rotation quaternion
        local_quat = torch.zeros_like(self.target_quat)
        local_quat[:, :3] = rotation_axis * sin_angle
        local_quat[:, 3:4] = cos_angle
        
        # Update the target orientation using quaternion multiplication
        self.target_quat = quat_mul(self.target_quat, local_quat)
        
        # Update target position based on linear velocity
        self.target_pos = self.target_pos + quat_rotate(self.target_quat, self.lin_vel * self.dt)

        # Apply the control based on method configuration
        if self.use_direct_pose_control:
            # Direct position and orientation setting
            self._direct_pose_control()
        else:
            # PD control method
            self._pd_control()

    def _direct_pose_control(self):
        """Set the position and orientation of the robot directly.
        
        This bypasses physics for testing and visualization.
        """
        # Set the root state directly
        # Decimate control updates for stability, only change position every few physics steps
        self.control_counter += 1
        if self.control_counter % self.control_decimation == 0:
            self.control_counter = 0
            
            # Prepare root state command tensor: [pos, rot, lin_vel, ang_vel]
            root_state_command = torch.zeros((self.num_envs, 13), device=self.device)
            root_state_command[:, 0:3] = self.target_pos
            root_state_command[:, 3:7] = self.target_quat
            root_state_command[:, 7:10] = self.lin_vel
            root_state_command[:, 10:13] = self.ang_vel
            
            # Set the root state directly
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(self.actor_indices),
                len(self.actor_indices)
            )

    def _pd_control(self):
        """Apply PD control to the base.
        
        Uses forces and torques to drive the base towards target pose.
        """
        # Calculate position error
        pos_error = self.target_pos - self.base_pos
        
        # Calculate orientation error
        quat_error = quat_conjugate(self.base_quat)
        quat_error = quat_mul(self.target_quat, quat_error)
        
        # Extract the axis-angle representation from the quaternion difference
        sin_half_angle = torch.norm(quat_error[:, :3], dim=1, keepdim=True)
        angle_error = 2.0 * torch.atan2(sin_half_angle, quat_error[:, 3:4])
        axis_error = quat_error[:, :3] / (sin_half_angle + 1e-8)
        orientation_error = axis_error * angle_error
        
        # Apply PD control
        # Forces for position control
        self.external_force = (
            self.cfg.control.position_p_gain * pos_error - 
            self.cfg.control.position_d_gain * self.base_lin_vel
        )
        
        # Torques for orientation control
        self.external_torque = (
            self.cfg.control.rotation_p_gain * orientation_error - 
            self.cfg.control.rotation_d_gain * self.base_ang_vel
        )
        
        # Apply the calculated forces and torques to the base
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.external_force),
            gymtorch.unwrap_tensor(self.base_pos),
            gymtorch.unwrap_tensor(self.root_body_indices),
            gymapi.ENV_SPACE
        )
        
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.external_torque),
            None,  # No force
            gymtorch.unwrap_tensor(self.root_body_indices),
            gymapi.LOCAL_SPACE
        )

    def compute_observations(self):
        """Compute the observations for the ElSpider Air base pose controller."""
        # Compute base observations: position, orientation, linear and angular velocity
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]
        
        # Normalize quaternion to avoid numerical issues
        self.base_quat = torch.where(
            self.base_quat[:, 3:4] < 0,
            -self.base_quat,
            self.base_quat
        )
        self.base_quat = self.base_quat / torch.norm(self.base_quat, dim=1, keepdim=True)
        
        # Create the observation vector
        obs_list = []
        
        # Common observations across the base
        # Base position
        obs_list.append(self.base_pos)
        # Base orientation (quaternion)
        obs_list.append(self.base_quat)
        # Base linear velocity
        obs_list.append(self.base_lin_vel)
        # Base angular velocity
        obs_list.append(self.base_ang_vel)
        
        # Add projected gravity (world frame gravity vector in base frame)
        proj_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        obs_list.append(proj_gravity)
        
        # Add command velocities
        obs_list.append(self.commands[:, :3])  # linear command
        obs_list.append(self.commands[:, 3:6])  # angular command
        
        # Concatenate all observations
        self.obs_buf = torch.cat(obs_list, dim=1)
        
        # Return the observation buffer
        return self.obs_buf

    def _update_ray_data(self):
        """Update raycasting data for each frame."""
        if hasattr(self, 'ray_caster') and self.cfg.raycaster.enable_raycast:
            # Set ray origins to the base position
            self.ray_caster.origins = self.base_pos.clone()
            
            # Cast rays and get updated ray data
            self.ray_caster.cast()

    def post_physics_step(self):
        """Process step after physics simulation."""
        # Update ray data for perception
        self._update_ray_data()
        
        # Process common post-physics steps
        super().post_physics_step() 