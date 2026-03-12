"""
Symmetry transformation functions for EL_4090_Safe environment.

This module provides observation and action augmentation via symmetry mirroring
for the EL_4090 hexapod robot with ATACOM safety layer.

Key features:
- Adaptive observation structure based on config (with/without height measurements)
- Support for u_mu feature vector (77 dims from ATACOM)
- Dynamic height measurement grid sizing
- Automatic detection of observation layout from environment config
"""

from typing import Tuple
import torch


@torch.no_grad()
def get_el4090_safe_xysym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, 
                                   env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply both left-right and back-forth symmetry transformations for EL_4090_Safe.
    
    Automatically detects observation structure based on env.cfg:
    - If measure_heights=True:  [core_66] + [height_meas] + [u_mu_77]
    - If measure_heights=False: [core_66] + [u_mu_77]
    
    Returns [batch*3, dim] where first batch is original, second is left-right mirrored, 
    third is back-forth mirrored.
    
    Observation core structure [0:66]:
        [0:3]     - base_lin_vel
        [3:6]     - base_ang_vel
        [6:9]     - projected_gravity
        [9:12]    - commands
        [12:30]   - dof_pos (18 DOFs)
        [30:48]   - dof_vel (18 DOFs)
        [48:66]   - previous actions (18 actions)
    
    u_mu structure (always at end, 77 dims):
        Safety layer output features for ATACOM
    
    Foot index order: LB(0-2), LF(3-5), LM(6-8), RB(9-11), RF(12-14), RM(15-17)
    Robot heading: x-axis forward, y-axis left, -y-axis right
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for config and terrain detection)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors [batch*3, dim]
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]
    
    if obs is not None:
        # ===== Detect observation structure from config =====
        core_obs_dim = 66  # Always: vel(3) + ang_vel(3) + gravity(3) + cmd(3) + dof_pos(18) + dof_vel(18) + act(18)
        u_mu_dim = 77     # Fixed for el_4090_safe
        
        measure_heights = False
        height_grid_x = 17
        height_grid_y = 11
        height_start_idx = core_obs_dim
        
        if env is not None and hasattr(env, 'cfg'):
            # Check if height measurements are enabled
            if hasattr(env.cfg.terrain, 'measure_heights'):
                measure_heights = env.cfg.terrain.measure_heights
            
            # Get grid dimensions if height measurements are enabled
            if measure_heights:
                if hasattr(env.cfg.terrain, 'measured_points_x'):
                    height_grid_x = len(env.cfg.terrain.measured_points_x)
                if hasattr(env.cfg.terrain, 'measured_points_y'):
                    height_grid_y = len(env.cfg.terrain.measured_points_y)
        
        # Calculate positions in observation vector
        height_meas_size = height_grid_x * height_grid_y if measure_heights else 0
        height_end_idx = height_start_idx + height_meas_size
        u_mu_start_idx = height_end_idx
        u_mu_end_idx = u_mu_start_idx + u_mu_dim
        
        # --- Left-Right Mirrored Observations ---
        obs_lr_mirrored = obs.clone()
        
        # Mirror core observation [0:66]
        # Mirror linear velocity y-component
        obs_lr_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_lr_mirrored[:, 3] = -obs[:, 3]
        obs_lr_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_lr_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_lr_mirrored[:, 10] = -obs[:, 10]
        obs_lr_mirrored[:, 11] = -obs[:, 11]
        
        # Swap left-right DOF positions (12:30)
        obs_lr_mirrored[:, 12:21] = obs[:, 21:30]  # Left legs get right leg positions
        obs_lr_mirrored[:, 21:30] = obs[:, 12:21]  # Right legs get left leg positions
        
        # Swap left-right DOF velocities (30:48)
        obs_lr_mirrored[:, 30:39] = obs[:, 39:48]
        obs_lr_mirrored[:, 39:48] = obs[:, 30:39]
        
        # Swap left-right previous actions (48:66)
        obs_lr_mirrored[:, 48:57] = obs[:, 57:66]
        obs_lr_mirrored[:, 57:66] = obs[:, 48:57]
        
        # Mirror height measurements along y-axis (if present)
        if measure_heights and obs.shape[1] > height_start_idx:
            for x in range(height_grid_x):
                for y in range(height_grid_y):
                    original_idx = height_start_idx + x * height_grid_y + y
                    mirrored_y = height_grid_y - y - 1
                    mirrored_idx = height_start_idx + x * height_grid_y + mirrored_y
                    obs_lr_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # u_mu (77 dims): Constraint slack variables
        # Structure (matches atacom.py constraint vector k):
        #   [0:36)   - Joint position limits (18 pairs of upper/lower, interleaved)
        #   [36:54)  - Joint velocity limits (18 dims)
        #   [54:72)  - Joint torque limits (18 dims)
        #   [72:74)  - Base height limits (2 dims: upper, lower)
        #   [74:77)  - Base tilt limits (3 dims: roll, pitch, yaw)
        
        if obs.shape[1] > u_mu_start_idx:
            # Joint position constraints [0:36) - swap left-right pairs
            # Structure: [q0_upper, q0_lower, q1_upper, q1_lower, ..., q17_upper, q17_lower]
            # Need to swap pairs: (0,1)<->(18,19), (2,3)<->(20,21), ..., (16,17)<->(34,35)
            for j in range(18):
                orig_upper = 2*j
                orig_lower = 2*j + 1
                mirror_upper = 2*(j+9) % 36  # Swap within 18-pair structure
                mirror_lower = 2*(j+9) % 36 + 1
                obs_lr_mirrored[:, u_mu_start_idx + orig_upper] = obs[:, u_mu_start_idx + mirror_upper]
                obs_lr_mirrored[:, u_mu_start_idx + orig_lower] = obs[:, u_mu_start_idx + mirror_lower]
            
            # Joint velocity constraints [36:54) - swap left-right
            obs_lr_mirrored[:, u_mu_start_idx + 36:u_mu_start_idx + 45] = obs[:, u_mu_start_idx + 45:u_mu_start_idx + 54]
            obs_lr_mirrored[:, u_mu_start_idx + 45:u_mu_start_idx + 54] = obs[:, u_mu_start_idx + 36:u_mu_start_idx + 45]
            
            # Joint torque constraints [54:72) - swap left-right
            obs_lr_mirrored[:, u_mu_start_idx + 54:u_mu_start_idx + 63] = obs[:, u_mu_start_idx + 63:u_mu_start_idx + 72]
            obs_lr_mirrored[:, u_mu_start_idx + 63:u_mu_start_idx + 72] = obs[:, u_mu_start_idx + 54:u_mu_start_idx + 63]
            
            # Base height constraints [72:74) - copy as-is (global constraint)
            obs_lr_mirrored[:, u_mu_start_idx + 72:u_mu_start_idx + 74] = obs[:, u_mu_start_idx + 72:u_mu_start_idx + 74]
            
            # Base tilt constraints [74:77) - mirror roll and pitch, negate yaw
            obs_lr_mirrored[:, u_mu_start_idx + 74] = obs[:, u_mu_start_idx + 74]   # roll unchanged (symmetric axis)
            obs_lr_mirrored[:, u_mu_start_idx + 75] = obs[:, u_mu_start_idx + 75]   # pitch unchanged
            obs_lr_mirrored[:, u_mu_start_idx + 76] = -obs[:, u_mu_start_idx + 76]  # yaw mirrored
        
        # --- Back-Forth Mirrored Observations ---
        obs_bf_mirrored = obs.clone()
        
        # Mirror core observation [0:66]
        # Mirror linear velocity x-component
        obs_bf_mirrored[:, 0] = -obs[:, 0]
        
        # Mirror angular velocity y and z components
        obs_bf_mirrored[:, 4] = -obs[:, 4]
        obs_bf_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity x-component
        obs_bf_mirrored[:, 6] = -obs[:, 6]
        
        # Mirror command velocities (x and angular z)
        obs_bf_mirrored[:, 9] = -obs[:, 9]
        obs_bf_mirrored[:, 11] = -obs[:, 11]
        
        # Swap back-front DOF positions (12:30): LB<->LF, RB<->RF, LM and RM unchanged
        obs_bf_mirrored[:, 12:15] = obs[:, 15:18]  # LB gets LF positions
        obs_bf_mirrored[:, 15:18] = obs[:, 12:15]  # LF gets LB positions
        obs_bf_mirrored[:, 21:24] = obs[:, 24:27]  # RB gets RF positions
        obs_bf_mirrored[:, 24:27] = obs[:, 21:24]  # RF gets RB positions
        
        # Swap back-front DOF velocities (30:48)
        obs_bf_mirrored[:, 30:33] = obs[:, 33:36]  # LB gets LF velocities
        obs_bf_mirrored[:, 33:36] = obs[:, 30:33]  # LF gets LB velocities
        obs_bf_mirrored[:, 39:42] = obs[:, 42:45]  # RB gets RF velocities
        obs_bf_mirrored[:, 42:45] = obs[:, 39:42]  # RF gets RB velocities
        
        # Swap back-front previous actions (48:66)
        obs_bf_mirrored[:, 48:51] = obs[:, 51:54]  # LB gets LF actions
        obs_bf_mirrored[:, 51:54] = obs[:, 48:51]  # LF gets LB actions
        obs_bf_mirrored[:, 57:60] = obs[:, 60:63]  # RB gets RF actions
        obs_bf_mirrored[:, 60:63] = obs[:, 57:60]  # RF gets RB actions
        
        # Mirror height measurements along x-axis (if present)
        if measure_heights and obs.shape[1] > height_start_idx:
            for x in range(height_grid_x):
                for y in range(height_grid_y):
                    original_idx = height_start_idx + x * height_grid_y + y
                    mirrored_x = height_grid_x - x - 1
                    mirrored_idx = height_start_idx + mirrored_x * height_grid_y + y
                    obs_bf_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # u_mu (77 dims): Constraint slack variables with back-forth transformation
        if obs.shape[1] > u_mu_start_idx:
            # Joint position constraints [0:36) - swap back-front pairs
            # LB(0-2)<->LF(3-5), RB(9-11)<->RF(12-14), LM(6-8) and RM(15-17) unchanged
            # In slack vars: pairs are interleaved, so (0,1)<->(3,4), (9,10)<->(12,13), etc.
            # LB: (0,1), LF: (3,4), LM: (6,7), RB: (9,10), RF: (12,13), RM: (15,16)
            for i in range(0, 2):  # LB <-> LF
                obs_bf_mirrored[:, u_mu_start_idx + i] = obs[:, u_mu_start_idx + 3 + i]
                obs_bf_mirrored[:, u_mu_start_idx + 3 + i] = obs[:, u_mu_start_idx + i]
            # LM unchanged
            for i in range(0, 2):  # RB <-> RF
                obs_bf_mirrored[:, u_mu_start_idx + 9 + i] = obs[:, u_mu_start_idx + 12 + i]
                obs_bf_mirrored[:, u_mu_start_idx + 12 + i] = obs[:, u_mu_start_idx + 9 + i]
            # RM unchanged
            # Copy rest (indices 2, 6-8, 15-17 which are single or unchanged legs)
            obs_bf_mirrored[:, u_mu_start_idx + 2] = obs[:, u_mu_start_idx + 2]
            obs_bf_mirrored[:, u_mu_start_idx + 6:9] = obs[:, u_mu_start_idx + 6:9]
            obs_bf_mirrored[:, u_mu_start_idx + 15:18] = obs[:, u_mu_start_idx + 15:18]
            
            # Joint velocity constraints [36:54) - swap back-front
            # Same mapping as core dof_vel: [0:3]LB<->[3:6]LF, [6:9]LM, [9:12]RB<->[12:15]RF, [15:18]RM
            obs_bf_mirrored[:, u_mu_start_idx + 36:39] = obs[:, u_mu_start_idx + 39:42]    # LB <-> LF
            obs_bf_mirrored[:, u_mu_start_idx + 39:42] = obs[:, u_mu_start_idx + 36:39]
            obs_bf_mirrored[:, u_mu_start_idx + 42:45] = obs[:, u_mu_start_idx + 42:45]    # LM unchanged
            obs_bf_mirrored[:, u_mu_start_idx + 45:48] = obs[:, u_mu_start_idx + 48:51]    # RB <-> RF
            obs_bf_mirrored[:, u_mu_start_idx + 48:51] = obs[:, u_mu_start_idx + 45:48]
            obs_bf_mirrored[:, u_mu_start_idx + 51:54] = obs[:, u_mu_start_idx + 51:54]    # RM unchanged
            
            # Joint torque constraints [54:72) - same mapping as velocity
            obs_bf_mirrored[:, u_mu_start_idx + 54:57] = obs[:, u_mu_start_idx + 57:60]    # LB <-> LF
            obs_bf_mirrored[:, u_mu_start_idx + 57:60] = obs[:, u_mu_start_idx + 54:57]
            obs_bf_mirrored[:, u_mu_start_idx + 60:63] = obs[:, u_mu_start_idx + 60:63]    # LM unchanged
            obs_bf_mirrored[:, u_mu_start_idx + 63:66] = obs[:, u_mu_start_idx + 66:69]    # RB <-> RF
            obs_bf_mirrored[:, u_mu_start_idx + 66:69] = obs[:, u_mu_start_idx + 63:66]
            obs_bf_mirrored[:, u_mu_start_idx + 69:72] = obs[:, u_mu_start_idx + 69:72]    # RM unchanged
            
            # Base height constraints [72:74) - copy as-is
            obs_bf_mirrored[:, u_mu_start_idx + 72:74] = obs[:, u_mu_start_idx + 72:74]
            
            # Base tilt constraints [74:77) - mirror pitch, keep roll and yaw
            obs_bf_mirrored[:, u_mu_start_idx + 74] = -obs[:, u_mu_start_idx + 74]  # roll mirrored
            obs_bf_mirrored[:, u_mu_start_idx + 75] = -obs[:, u_mu_start_idx + 75]  # pitch mirrored
            obs_bf_mirrored[:, u_mu_start_idx + 76] = obs[:, u_mu_start_idx + 76]   # yaw unchanged
        
        # Combine original, left-right mirrored, and back-forth mirrored observations
        obs_augmented = torch.cat([obs, obs_lr_mirrored, obs_bf_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # --- Left-Right Mirrored Actions ---
        # Foot index: LB(0-2), LF(3-5), LM(6-8), RB(9-11), RF(12-14), RM(15-17)
        actions_lr_mirrored = actions.clone()
        
        # Swap left and right legs
        actions_lr_mirrored[:, 0:9] = actions[:, 9:18]   # Left legs get right leg actions
        actions_lr_mirrored[:, 9:18] = actions[:, 0:9]   # Right legs get left leg actions
        
        # --- Back-Forth Mirrored Actions ---
        actions_bf_mirrored = actions.clone()
        
        # Swap back and front legs: LB<->LF, RB<->RF, LM and RM stay as-is
        actions_bf_mirrored[:, 0:3] = actions[:, 3:6]    # LB gets LF actions
        actions_bf_mirrored[:, 3:6] = actions[:, 0:3]    # LF gets LB actions
        actions_bf_mirrored[:, 9:12] = actions[:, 12:15] # RB gets RF actions
        actions_bf_mirrored[:, 12:15] = actions[:, 9:12] # RF gets RB actions
        
        # Combine original, left-right mirrored, and back-forth mirrored actions
        actions_augmented = torch.cat([actions, actions_lr_mirrored, actions_bf_mirrored], dim=0)
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented


@torch.no_grad()
def get_el4090_safe_xsym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, 
                                  env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply left-right symmetry transformation for EL_4090_Safe.
    
    Automatically detects observation structure based on env.cfg:
    - If measure_heights=True:  [core_66] + [height_meas] + [u_mu_77]
    - If measure_heights=False: [core_66] + [u_mu_77]
    
    Returns [batch*2, dim] where first batch is original, second is left-right mirrored.
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for config and terrain detection)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors [batch*2, dim]
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]
    
    if obs is not None:
        # ===== Detect observation structure from config =====
        core_obs_dim = 66
        u_mu_dim = 77
        
        measure_heights = False
        height_grid_x = 17
        height_grid_y = 11
        height_start_idx = core_obs_dim
        
        if env is not None and hasattr(env, 'cfg'):
            # Check if height measurements are enabled
            if hasattr(env.cfg.terrain, 'measure_heights'):
                measure_heights = env.cfg.terrain.measure_heights
            
            # Get grid dimensions if height measurements are enabled
            if measure_heights:
                if hasattr(env.cfg.terrain, 'measured_points_x'):
                    height_grid_x = len(env.cfg.terrain.measured_points_x)
                if hasattr(env.cfg.terrain, 'measured_points_y'):
                    height_grid_y = len(env.cfg.terrain.measured_points_y)
        
        # Calculate positions in observation vector
        height_meas_size = height_grid_x * height_grid_y if measure_heights else 0
        height_end_idx = height_start_idx + height_meas_size
        u_mu_start_idx = height_end_idx
        u_mu_end_idx = u_mu_start_idx + u_mu_dim
        
        # Create left-right mirrored observations
        obs_mirrored = obs.clone()
        
        # Mirror linear velocity y-component
        obs_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_mirrored[:, 3] = -obs[:, 3]
        obs_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_mirrored[:, 10] = -obs[:, 10]
        obs_mirrored[:, 11] = -obs[:, 11]
        
        # Swap left-right DOF positions (12:30)
        obs_mirrored[:, 12:21] = obs[:, 21:30]  # Left legs get right leg positions
        obs_mirrored[:, 21:30] = obs[:, 12:21]  # Right legs get left leg positions
        
        # Swap left-right DOF velocities (30:48)
        obs_mirrored[:, 30:39] = obs[:, 39:48]
        obs_mirrored[:, 39:48] = obs[:, 30:39]
        
        # Swap left-right previous actions (48:66)
        obs_mirrored[:, 48:57] = obs[:, 57:66]
        obs_mirrored[:, 57:66] = obs[:, 48:57]
        
        # Mirror height measurements along y-axis (if present)
        if measure_heights and obs.shape[1] > height_start_idx:
            for x in range(height_grid_x):
                for y in range(height_grid_y):
                    original_idx = height_start_idx + x * height_grid_y + y
                    mirrored_y = height_grid_y - y - 1
                    mirrored_idx = height_start_idx + x * height_grid_y + mirrored_y
                    obs_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # u_mu (77 dims): Constraint slack variables with left-right transformation
        if obs.shape[1] > u_mu_start_idx:
            # Joint position constraints [0:36) - swap left-right pairs
            for j in range(18):
                orig_upper = 2*j
                orig_lower = 2*j + 1
                mirror_upper = 2*(j+9) % 36
                mirror_lower = 2*(j+9) % 36 + 1
                obs_mirrored[:, u_mu_start_idx + orig_upper] = obs[:, u_mu_start_idx + mirror_upper]
                obs_mirrored[:, u_mu_start_idx + orig_lower] = obs[:, u_mu_start_idx + mirror_lower]
            
            # Joint velocity constraints [36:54) - swap left-right
            obs_mirrored[:, u_mu_start_idx + 36:u_mu_start_idx + 45] = obs[:, u_mu_start_idx + 45:u_mu_start_idx + 54]
            obs_mirrored[:, u_mu_start_idx + 45:u_mu_start_idx + 54] = obs[:, u_mu_start_idx + 36:u_mu_start_idx + 45]
            
            # Joint torque constraints [54:72) - swap left-right
            obs_mirrored[:, u_mu_start_idx + 54:u_mu_start_idx + 63] = obs[:, u_mu_start_idx + 63:u_mu_start_idx + 72]
            obs_mirrored[:, u_mu_start_idx + 63:u_mu_start_idx + 72] = obs[:, u_mu_start_idx + 54:u_mu_start_idx + 63]
            
            # Base height constraints [72:74) - copy as-is (global constraint)
            obs_mirrored[:, u_mu_start_idx + 72:u_mu_start_idx + 74] = obs[:, u_mu_start_idx + 72:u_mu_start_idx + 74]
            
            # Base tilt constraints [74:77) - mirror roll and pitch, negate yaw
            obs_mirrored[:, u_mu_start_idx + 74] = obs[:, u_mu_start_idx + 74]   # roll unchanged
            obs_mirrored[:, u_mu_start_idx + 75] = obs[:, u_mu_start_idx + 75]   # pitch unchanged
            obs_mirrored[:, u_mu_start_idx + 76] = -obs[:, u_mu_start_idx + 76]  # yaw mirrored
        
        # Combine original and mirrored observations
        obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # Mirror the actions - swap left and right legs
        # EL_4090 has 18 actions (6 legs × 3 joints)
        actions_mirrored = actions.clone()
        
        actions_mirrored[:, 0:9] = actions[:, 9:18]   # Left legs get right leg actions
        actions_mirrored[:, 9:18] = actions[:, 0:9]   # Right legs get left leg actions
        
        # Combine original and mirrored actions
        actions_augmented = torch.cat([actions, actions_mirrored], dim=0)
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented
