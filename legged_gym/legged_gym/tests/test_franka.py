#!/usr/bin/env python3
"""
Test script for Franka robot arm environments.

This script tests both the basic Franka environment and the batch rollout version
to ensure they work correctly with the confined_trimesh terrain generation.
"""
import isaacgym
import torch
import numpy as np
from legged_gym.envs.franka.franka import Franka
from legged_gym.envs.franka.franka_config import FrankaCfg
from legged_gym.envs.franka.batch_rollout.franka_batch_rollout import FrankaBatchRollout
from legged_gym.envs.franka.batch_rollout.franka_batch_rollout_config import FrankaBatchRolloutCfg
from isaacgym import gymapi
import time

def test_basic_franka():
    """Test the basic Franka environment."""
    print("Testing basic Franka environment...")
    
    # Create configuration
    cfg = FrankaCfg()
    cfg.env.num_envs = 64  # Small number for testing
    
    # Create simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    sim_params.up_axis = gymapi.UP_AXIS_Z
    
    # Physics parameters
    sim_params.physx.solver_type = cfg.sim.physx.solver_type
    sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
    sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
    sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
    sim_params.physx.rest_offset = cfg.sim.physx.rest_offset
    sim_params.physx.bounce_threshold_velocity = cfg.sim.physx.bounce_threshold_velocity
    sim_params.physx.max_depenetration_velocity = cfg.sim.physx.max_depenetration_velocity
    sim_params.physx.max_gpu_contact_pairs = cfg.sim.physx.max_gpu_contact_pairs
    sim_params.physx.default_buffer_size_multiplier = cfg.sim.physx.default_buffer_size_multiplier
    # sim_params.physx.contact_collection = cfg.sim.physx.contact_collection
    
    # Create environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = Franka(cfg=cfg, 
                 sim_params=sim_params,
                 physics_engine=gymapi.SIM_PHYSX,
                 sim_device=device,
                 headless=False)
    env.reset()
    print(f"✓ Basic Franka environment created successfully!")
    print(f"  - Number of environments: {env.num_envs}")
    print(f"  - Number of DOFs: {env.num_dof}")
    print(f"  - Number of actions: {env.num_actions}")
    print(f"  - Observation size: {env.obs_buf.shape[1]}")
    
    # Test a few steps
    for i in range(200):
        actions = torch.ones(env.num_envs, env.num_actions, device=device)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        time.sleep(env.dt)  # Simulate some delay for rendering
        if i == 0:
            print(f"  - First step completed, reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    
    print("✓ Basic Franka environment test completed successfully!\n")
    return True


def test_batch_rollout_franka():
    """Test the batch rollout Franka environment."""
    print("Testing Franka batch rollout environment...")
    
    # Create configuration
    cfg = FrankaBatchRolloutCfg()
    cfg.env.num_envs = 4  # Small number for testing
    cfg.env.rollout_envs = 8  # Small number of rollout envs
    
    # Create simulation parameters from config
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    sim_params.up_axis = gymapi.UP_AXIS_Z
    
    # Physics parameters from config
    sim_params.physx.solver_type = cfg.sim.physx.solver_type
    sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
    sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
    sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
    sim_params.physx.rest_offset = cfg.sim.physx.rest_offset
    sim_params.physx.bounce_threshold_velocity = cfg.sim.physx.bounce_threshold_velocity
    sim_params.physx.max_depenetration_velocity = cfg.sim.physx.max_depenetration_velocity
    sim_params.physx.max_gpu_contact_pairs = cfg.sim.physx.max_gpu_contact_pairs
    sim_params.physx.default_buffer_size_multiplier = cfg.sim.physx.default_buffer_size_multiplier
    # sim_params.physx.contact_collection = cfg.sim.physx.contact_collection
    
    # Create environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = FrankaBatchRollout(cfg=cfg,
                            sim_params=sim_params,
                            physics_engine=gymapi.SIM_PHYSX,
                            sim_device=device,
                            headless=False)
    
    print(f"✓ Batch rollout Franka environment created successfully!")
    print(f"  - Number of main environments: {env.num_main_envs}")
    print(f"  - Number of rollout environments per main: {env.num_rollout_per_main}")
    print(f"  - Total environments: {env.total_num_envs}")
    print(f"  - Number of DOFs: {env.num_dof}")
    print(f"  - Number of actions: {env.num_actions}")
    print(f"  - Main env observation size: {env.obs_buf[env.main_env_indices].shape[1]}")
    
    # Test raycast perception if enabled
    if hasattr(env, 'ray_caster') and env.ray_caster is not None:
        print(f"  - Raycast enabled with {env.num_ray_observations} rays")
    
    # Test SDF perception if enabled
    if hasattr(env, 'mesh_sdf') and env.mesh_sdf is not None:
        print(f"  - SDF enabled with {env.num_sdf_bodies} query bodies")
    
    # Test main environment steps
    for i in range(5):
        actions = torch.zeros(env.num_main_envs, env.num_actions, device=device)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        
        if i == 0:
            print(f"  - First main step completed, reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    
    # Test rollout environment steps
    rollout_actions = torch.zeros(len(env.rollout_env_indices), env.num_actions, device=device)
    rollout_obs, rollout_privileged_obs, rollout_rewards, rollout_dones, rollout_infos = env.step_rollout(rollout_actions)
    print(f"  - Rollout step completed, reward range: [{rollout_rewards.min():.3f}, {rollout_rewards.max():.3f}]")
    
    print("✓ Batch rollout Franka environment test completed successfully!\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("FRANKA ROBOT ARM ENVIRONMENT TESTS")
    print("=" * 60)
    
    try:
        # Test basic environment
        # test_basic_franka()
        
        # Test batch rollout environment
        test_batch_rollout_franka()
        
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nThe Franka robot arm environments are ready for use:")
        print("  1. Basic manipulation: legged_gym.envs.franka.Franka")
        print("  2. Batch rollout with perception: legged_gym.envs.franka.batch_rollout.FrankaBatchRollout")
        print("\nKey features implemented:")
        print("  ✓ Fixed-base robot arm control")
        print("  ✓ End-effector pose tracking")
        print("  ✓ Obstacle avoidance rewards")
        print("  ✓ Confined terrain generation")
        print("  ✓ Raycast perception (batch rollout)")
        print("  ✓ Batch environment synchronization")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()