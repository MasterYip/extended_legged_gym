import os
import sys
import argparse
import isaacgym
import torch
import numpy as np
from datetime import datetime
from isaacgym import gymapi, gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, get_default_args, task_registry
from legged_gym.utils.helpers import parse_default_arguments, class_to_dict


def get_terrain_estimator_play_args():
    """Parse command line arguments for terrain estimator play."""
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat",
            "help": "Task name for environment with depth camera and raycast support"},
        {"name": "--resume", "action": "store_true", "default": False, 
            "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, 
            "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, 
            "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "default": "-1",
            "help": "Name of the run to load when resume=True. If -1: will load the last run."},
        {"name": "--checkpoint", "type": int, "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint."},
        
        {"name": "--headless", "action": "store_true", "default": False, 
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, 
            "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 16,
            "help": "Number of environments to create"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, 
            "help": "Maximum number of training iterations"},
        
        # Play specific arguments
        {"name": "--num_episodes", "type": int, "default": 10,
            "help": "Number of episodes to play in the environment"},
        {"name": "--pretrained_policy", "type": str, "default": None,
            "help": "Path to pretrained policy for environment control"},
    ]
    
    # Use the same argument parsing as terrain_est_train.py
    try:
        from isaacgym import gymutil
        args = gymutil.parse_arguments(
            description="Terrain Estimator Play",
            custom_parameters=custom_parameters
        )
    except:
        # Fallback to argparse if gymutil is not available
        parser = argparse.ArgumentParser(description="Terrain Estimator Play")
        for param in custom_parameters:
            parser.add_argument(param["name"], **{k: v for k, v in param.items() if k != "name"})
        args = parser.parse_args()
    
    # Device alignment (same as terrain_est_train.py)
    if hasattr(args, 'compute_device_id'):
        args.sim_device_id = args.compute_device_id
        args.sim_device = args.sim_device_type
        if args.sim_device == 'cuda':
            args.sim_device += f":{args.sim_device_id}"
    
    return args


def play_terrain_estimator(args):
    """Play terrain estimator with visualization."""
    
    # Import RSL-RL
    try:
        from rsl_rl.runners import TerrainEstimatorRunner
    except ImportError as e:
        print(f"Failed to import rsl_rl: {e}")
        print("Please install rsl_rl or check your Python path")
        return
    
    # Configure device
    device = args.rl_device if torch.cuda.is_available() and 'cuda' in args.rl_device else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override some environment settings for better visualization
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)  # Limit environments for better visualization
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 0
    
    
    # Determine log directory and model path first to inspect checkpoint
    if args.load_run == '-1':
        # Get the latest run
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'terrain_est_elspider_air')
        if os.path.exists(log_root):
            runs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
            if runs:
                runs.sort()
                load_run = runs[-1]
                log_dir = os.path.join(log_root, load_run)
            else:
                print(f"No previous runs found in {log_root}")
                return
        else:
            print(f"Log directory {log_root} does not exist")
            return
    else:
        # Use specified run
        log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'terrain_estimator', args.task, args.load_run)
    
    # Get model path
    if args.checkpoint == -1:
        # Load latest checkpoint
        model_files = [f for f in os.listdir(log_dir) if f.startswith('estimator_') and f.endswith('.pt')]
        if not model_files:
            print(f"No estimator checkpoints found in {log_dir}")
            return
        
        # Sort by iteration number
        model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_path = os.path.join(log_dir, model_files[-1])
    else:
        model_path = os.path.join(log_dir, f'estimator_{args.checkpoint}.pt')
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return
    
    print(f"Loading terrain estimator from: {model_path}")
    
    # Load checkpoint to inspect model architecture
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect the number of raycast outputs from the final decoder layer
        decoder_weight_key = None
        for key in model_state_dict.keys():
            if 'decoder' in key and 'weight' in key:
                decoder_weight_key = key
        
        if decoder_weight_key:
            # Get the output dimension from the last decoder layer
            decoder_weights = model_state_dict[decoder_weight_key]
            detected_raycast_outputs = decoder_weights.shape[0]
            print(f"Detected raycast outputs from checkpoint: {detected_raycast_outputs}")
            
            # Override the environment raycast configuration to match checkpoint
            if hasattr(env_cfg, 'raycaster'):
                env_cfg.raycaster.num_rays = detected_raycast_outputs
                print(f"Updated environment raycast configuration to match checkpoint: {detected_raycast_outputs} rays")
        else:
            print("Warning: Could not detect raycast output dimension from checkpoint")
            
    except Exception as e:
        print(f"Warning: Could not inspect checkpoint architecture: {e}")
    
    # Create environment after updating configuration
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    train_cfg = class_to_dict(train_cfg)  # Convert to dict for easier manipulation
    
    # Configure training for terrain estimator (same structure as terrain_est_train.py)
    if 'estimator' not in train_cfg:
        train_cfg['estimator'] = {
            "encoder_output_dim": 64,
            "memory_hidden_size": 256,
            "memory_num_layers": 1,
            "memory_type": "gru",
            "decoder_hidden_dims": [128, 64],
            "activation": "elu",
        }
    
    if 'algorithm' not in train_cfg:
        train_cfg['algorithm'] = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'loss_type': 'mse',
        }
    
    if 'policy' not in train_cfg:
        train_cfg['policy'] = {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }
    
    # Configure runner settings for play mode
    train_cfg['runner']['enable_visualization'] = True
    train_cfg['runner']['visualization_interval'] = 1
    train_cfg['runner']['num_steps_per_env'] = 24
    
    # Find pretrained policy path for environment control
    pretrained_policy_path = None
    if hasattr(args, 'pretrained_policy') and args.pretrained_policy:
        pretrained_policy_path = args.pretrained_policy
    else:
        # Try to find a policy in the same log directory or a related policy directory
        policy_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task)
        if os.path.exists(policy_log_root):
            policy_runs = [d for d in os.listdir(policy_log_root) if os.path.isdir(os.path.join(policy_log_root, d))]
            if policy_runs:
                policy_runs.sort()
                policy_dir = os.path.join(policy_log_root, policy_runs[-1])
                policy_files = [f for f in os.listdir(policy_dir) if f.startswith('model_') and f.endswith('.pt')]
                if policy_files:
                    policy_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                    pretrained_policy_path = os.path.join(policy_dir, policy_files[-1])
                    print(f"Found pretrained policy: {pretrained_policy_path}")
    
    # Create terrain estimator runner
    runner = TerrainEstimatorRunner(
        env=env,
        train_cfg=train_cfg,
        pretrained_policy_path=pretrained_policy_path,
        log_dir=None,  # No logging during play
        device=device
    )
    
    # Load trained estimator
    try:
        runner.load(model_path, load_optimizer=False)
        print(f"Successfully loaded terrain estimator from iteration {runner.current_learning_iteration}")
    except Exception as e:
        print(f"Failed to load terrain estimator: {e}")
        print("This might be due to architecture mismatch. Please check:")
        print("1. The raycast configuration in your environment matches the training configuration")
        print("2. The estimator architecture configuration matches the training configuration")
        return
    
    # Start play mode
    print("\n" + "="*50)
    print("TERRAIN ESTIMATOR PLAY MODE")
    print("="*50)
    print("Controls:")
    print("  - Press 'v' to toggle debug visualization")
    print("  - Press 'c' to clear visualization")
    print("  - Press ESC to exit")
    print("  - Red points: Ground truth raycast")
    print("  - Blue points: Estimator predictions")
    print("="*50 + "\n")
    
    try:
        runner.play(
            num_episodes=args.num_episodes,
            deterministic=True,
            enable_visualization=True
        )
    except KeyboardInterrupt:
        print("\nPlay interrupted by user")
    except Exception as e:
        print(f"Error during play: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        env.close()


if __name__ == '__main__':
    args = get_terrain_estimator_play_args()
    play_terrain_estimator(args)
