# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for terrain estimator using supervised learning from depth images to raycast data.
"""

import numpy as np
import os
from datetime import datetime
import argparse

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import task_registry
from rsl_rl.runners import TerrainEstimatorRunner
import torch


def get_terrain_estimator_args():
    """Parse command line arguments for terrain estimator training."""
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_rough_raycast",
            "help": "Task name for environment with depth camera and raycast support"},
        {"name": "--resume", "action": "store_true", "default": False, 
            "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, 
            "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, 
            "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run."},
        {"name": "--checkpoint", "type": int, "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint."},
        {"name": "--pretrained_policy", "type": str,
            "help": "Path to pretrained policy for robot control during data collection"},
        
        {"name": "--headless", "action": "store_true", "default": False, 
            "help": "Force display off at all times"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 4096,
            "help": "Number of environments to create"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 1000,
            "help": "Maximum number of training iterations"},
        
        # Terrain estimator specific arguments
        {"name": "--learning_rate", "type": float, "default": 1e-3,
            "help": "Learning rate for terrain estimator"},
        {"name": "--loss_type", "type": str, "default": "mse", 
            "choices": ["mse", "huber", "l1"], "help": "Loss function type"},
        {"name": "--num_learning_epochs", "type": int, "default": 5,
            "help": "Number of learning epochs per iteration"},
        {"name": "--gradient_length", "type": int, "default": 15,
            "help": "Gradient accumulation length"},
        {"name": "--num_steps_per_env", "type": int, "default": 24,
            "help": "Number of steps per environment per iteration"},
        {"name": "--save_interval", "type": int, "default": 100,
            "help": "Save model every N iterations"},
        {"name": "--enable_visualization", "action": "store_true", "default": False,
            "help": "Enable visualization of estimator predictions vs ground truth"},
        {"name": "--visualization_interval", "type": int, "default": 1,
            "help": "Visualize every N iterations (default: 50)"},
    ]
    
    # Use the same argument parsing as the main legged_gym
    try:
        from isaacgym import gymutil
        args = gymutil.parse_arguments(
            description="Terrain Estimator Training",
            custom_parameters=custom_parameters
        )
    except:
        # Fallback to argparse if gymutil is not available
        parser = argparse.ArgumentParser(description="Terrain Estimator Training")
        for param in custom_parameters:
            parser.add_argument(param["name"], **{k: v for k, v in param.items() if k != "name"})
        args = parser.parse_args()
    
    # Device alignment
    if hasattr(args, 'compute_device_id'):
        args.sim_device_id = args.compute_device_id
        args.sim_device = args.sim_device_type
        if args.sim_device == 'cuda':
            args.sim_device += f":{args.sim_device_id}"
    
    return args

# FIXME: part of this should come from the config file
def create_terrain_estimator_config(args):
    """Create configuration for terrain estimator training."""
    train_cfg = {
        "runner": {
            "num_steps_per_env": args.num_steps_per_env,
            "save_interval": args.save_interval,
            "logger": "tensorboard",
            "max_iterations": args.max_iterations,
            "policy_class_name": "ActorCritic",  # Default to MLP policy
            "enable_visualization": args.enable_visualization,
            "visualization_interval": args.visualization_interval,
        },
        "algorithm": {
            "num_learning_epochs": args.num_learning_epochs,
            "gradient_length": args.gradient_length,
            "learning_rate": args.learning_rate,
            "max_grad_norm": 1.0,
            "loss_type": args.loss_type,
        },
        "estimator": {
            "encoder_output_dim": 64,
            "memory_hidden_size": 256,
            "memory_num_layers": 1,
            "memory_type": "gru",
            "decoder_hidden_dims": [128, 64],
            "activation": "elu",
        },
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }
    }
    return train_cfg


def setup_environment_for_terrain_estimation(env_cfg):
    """Configure environment for terrain estimator training."""
    # Enable depth camera
    if hasattr(env_cfg, 'depth'):
        env_cfg.depth.use_camera = True
        env_cfg.depth.update_interval = 1  # Update every physics step
        env_cfg.depth.buffer_len = 1  # Single frame for now
        print("Enabled depth camera for terrain estimation")
    else:
        print("Warning: Environment does not support depth camera")
    
    # Enable raycast
    if hasattr(env_cfg, 'raycaster'):
        env_cfg.raycaster.enable_raycast = True
        print(f"Enabled raycast with {getattr(env_cfg.raycaster, 'num_rays', 'unknown')} rays")
    else:
        print("Warning: Environment does not support raycast")
    
    return env_cfg


def train_terrain_estimator(args):
    """Main training function for terrain estimator."""
    print(f"Training terrain estimator for task: {args.task}")
    print(f"Using device: {args.rl_device}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    
    # Get environment configuration
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    
    # Override with command line arguments
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.env.seed = args.seed
    
    # Setup environment for terrain estimation
    env_cfg = setup_environment_for_terrain_estimation(env_cfg)
    
    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Create terrain estimator training configuration
    train_cfg = create_terrain_estimator_config(args)
    
    # Create log directory
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f"terrain_estimator_{args.task}"
    
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dirs = os.listdir(os.path.join("logs", experiment_name))
    log_dirs.sort()
    last_log_dir = os.path.join("logs", experiment_name, log_dirs[-1])

    log_dir = os.path.join("logs", experiment_name, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Create terrain estimator runner
    runner = TerrainEstimatorRunner(
        env=env,
        train_cfg=train_cfg,
        pretrained_policy_path=args.pretrained_policy if hasattr(args, 'pretrained_policy') else None,
        log_dir=log_dir,
        device=args.rl_device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        if args.load_run and args.load_run != "-1":
            load_path = os.path.join("logs", experiment_name, args.load_run)
        else:
            load_path = last_log_dir
        
        if args.checkpoint and args.checkpoint != -1:
            checkpoint_file = f"estimator_{args.checkpoint}.pt"
        else:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(load_path) if f.startswith("estimator_") and f.endswith(".pt")]
            if checkpoints:
                checkpoint_numbers = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
                latest_checkpoint = max(checkpoint_numbers)
                checkpoint_file = f"estimator_{latest_checkpoint}.pt"
            else:
                print("No checkpoints found for resuming")
                checkpoint_file = None
        
        if checkpoint_file:
            checkpoint_path = os.path.join(load_path, checkpoint_file)
            if os.path.exists(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                runner.load(checkpoint_path)
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
    
    # Train the terrain estimator
    print("Starting terrain estimator training...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("Training completed!")


if __name__ == '__main__':
    args = get_terrain_estimator_args()
    
    # Set up GPU memory recording if needed
    record_gpu_memory = False
    if record_gpu_memory:
        print("Recording GPU memory usage")
        torch.cuda.memory._record_memory_history()
    
    try:
        train_terrain_estimator(args)
    except KeyboardInterrupt as e:
        print(f"Training interrupted: {e}")
    
    if record_gpu_memory:
        torch.cuda.memory._dump_snapshot("terrain_est_gpumem_snap.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Memory history saved to terrain_est_gpumem_snap.pickle")


