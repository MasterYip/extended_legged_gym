# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from typing import Optional
from collections import deque

import rsl_rl
from rsl_rl.algorithms import EstimatorDistillation
from rsl_rl.modules import TerrainEstimator
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state


class TerrainEstimatorRunner:
    """Runner for training terrain estimator using distillation from raycast ground truth."""

    def __init__(self, env: VecEnv, train_cfg: dict, pretrained_policy_path: str = None, log_dir: Optional[str] = None, device="cpu"):
        self.device = device
        self.env = env
        self.log_dir = log_dir
        self.pretrained_policy_path = pretrained_policy_path

        # Parse configuration
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.estimator_cfg = train_cfg["estimator"]
        self.policy_cfg = train_cfg["policy"]

        # Configure multi-GPU if available
        self._configure_multi_gpu()

        # Setup estimator dimensions
        self._setup_dimensions()

        # Initialize terrain estimator
        self.estimator = self._initialize_estimator()

        # Initialize algorithm (EstimatorDistillation)
        self.alg = self._initialize_algorithm()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.enable_visualization = self.cfg.get("enable_visualization", False)
        self.visualization_interval = self.cfg.get("visualization_interval", 1)  # Visualize every N iterations

        # Initialize storage
        self._initialize_storage()

        # Setup logging
        self._setup_logging()
        
        self.env.reset()

        # Load pretrained policy for environment rollouts
        if pretrained_policy_path and os.path.exists(pretrained_policy_path):
            self._load_pretrained_policy(pretrained_policy_path, self.num_obs, self.num_privileged_obs)
        else:
            print("Warning: No pretrained policy provided. Environment will use random actions.")

    def _setup_dimensions(self):
        """Setup dimensions for estimator training."""
        # Get depth image dimensions
        if hasattr(self.env, 'get_depth_images'):
            # Try to get a sample depth image to determine dimensions
            sample_depth = self.env.get_depth_images()
            if sample_depth is not None:
                self.depth_image_shape = sample_depth.shape[-2:]  # (height, width)
            else:
                # Default depth image size if not available
                self.depth_image_shape = (58, 87)  # Common depth camera resolution
        else:
            # Default depth image size
            self.depth_image_shape = (58, 87)

        # Proprioceptive data dimension (base linear velocity + angular velocity)
        self.proprio_dim = 6  # 3 for lin_vel + 3 for ang_vel

        # Raycast output dimension
        if hasattr(self.env, 'num_ray_observations'):
            self.num_raycast_outputs = self.env.num_ray_observations
        else:
            # Try to get from configuration
            if hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'raycaster'):
                self.num_raycast_outputs = getattr(self.env.cfg.raycaster, 'num_rays', 32)
            else:
                self.num_raycast_outputs = 32  # Default

        # Environment observation dimensions for pretrained policy
        if hasattr(self.env, 'num_obs'):
            self.num_obs = self.env.num_obs
            self.num_privileged_obs = getattr(self.env, 'num_privileged_obs', self.num_obs)
        else:
            # Try to get observation dimensions from initial observation
            try:
                obs, extras = self.env.get_observations()
                self.num_obs = obs.shape[1]
                # Check if there are privileged observations
                if "critic" in extras.get("observations", {}):
                    self.num_privileged_obs = extras["observations"]["critic"].shape[1]
                else:
                    self.num_privileged_obs = self.num_obs
            except:
                # Fallback defaults
                self.num_obs = 48  # Common observation size
                self.num_privileged_obs = self.num_obs

        print(f"Estimator dimensions:")
        print(f"  Depth image shape: {self.depth_image_shape}")
        print(f"  Proprioceptive dim: {self.proprio_dim}")
        print(f"  Raycast outputs: {self.num_raycast_outputs}")
        print(f"  Environment obs dim: {self.num_obs}")
        print(f"  Privileged obs dim: {self.num_privileged_obs}")

    def _initialize_estimator(self):
        """Initialize the terrain estimator model."""
        estimator = TerrainEstimator(
            depth_image_shape=self.depth_image_shape,
            proprio_dim=self.proprio_dim,
            num_raycast_outputs=self.num_raycast_outputs,
            **self.estimator_cfg
        ).to(self.device)

        return estimator

    def _initialize_algorithm(self):
        """Initialize the EstimatorDistillation algorithm."""
        alg_kwargs = dict(self.alg_cfg)
        if hasattr(self, 'multi_gpu_cfg') and self.multi_gpu_cfg:
            alg_kwargs["multi_gpu_cfg"] = self.multi_gpu_cfg

        return EstimatorDistillation(self.estimator, device=self.device, **alg_kwargs)

    def _initialize_storage(self):
        """Initialize rollout storage."""
        self.alg.init_storage(
            "estimation",  # training type
            self.env.num_envs,
            self.num_steps_per_env,
            self.depth_image_shape,
            [self.proprio_dim],
            [self.num_raycast_outputs],
        )

    def _setup_logging(self):
        """Setup logging configuration."""
        self.disable_logs = hasattr(self, 'is_distributed') and self.is_distributed and self.gpu_global_rank != 0
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _load_pretrained_policy(self, policy_path, num_obs, num_privileged_obs=None):
        """Load pretrained policy for environment rollouts.
        
        Args:
            policy_path: Path to the policy checkpoint
            num_obs: Number of observation dimensions
            num_privileged_obs: Number of privileged observation dimensions
        """
        try:
            print(f"Loading pretrained policy from {policy_path}")
            
            # Check if policy file exists
            if not os.path.exists(policy_path):
                raise FileNotFoundError(f"Policy checkpoint not found at {policy_path}")
            
            # Directly load checkpoint
            checkpoint = torch.load(policy_path, map_location=self.device)
            
            # Use provided observation dimensions or fallback to num_obs
            if num_privileged_obs is None:
                num_privileged_obs = num_obs
            
            # Determine if policy is recurrent based on policy class name
            policy_class_name = self.cfg.get("policy_class_name", "ActorCritic")
            is_recurrent = "Recurrent" in policy_class_name
            
            # Create the appropriate actor_critic model using configuration
            from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
            
            if is_recurrent:
                print("Creating recurrent actor-critic model")
                actor_critic = ActorCriticRecurrent(
                    num_obs,
                    num_privileged_obs,
                    self.env.num_actions,
                    actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
                    critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
                    activation=self.policy_cfg.get("activation", "elu"),
                ).to(self.device)
            else:
                print("Creating MLP actor-critic model")
                actor_critic = ActorCritic(
                    num_obs,
                    num_privileged_obs,
                    self.env.num_actions,
                    actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
                    critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
                    activation=self.policy_cfg.get("activation", "elu"),
                ).to(self.device)
            
            # Load state dict from checkpoint
            if "model_state_dict" in checkpoint:
                actor_critic.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded model from 'model_state_dict'")
            elif "actor_state_dict" in checkpoint:
                actor_critic.actor.load_state_dict(checkpoint["actor_state_dict"])
                print("Loaded model from 'actor_state_dict'")
            else:
                raise ValueError("Unsupported checkpoint format, couldn't find model_state_dict or actor_state_dict")
            
            # Set to evaluation mode
            actor_critic.eval()
            
            # Store the policy
            self.pretrained_policy = actor_critic
            self.obs_mean = None 
            self.obs_var = None
            
            # Check for observation normalization parameters
            if "obs_mean" in checkpoint and "obs_var" in checkpoint:
                self.obs_mean = checkpoint["obs_mean"].to(self.device)
                self.obs_var = checkpoint["obs_var"].to(self.device)
                print("Loaded observation standardization parameters from checkpoint")
            else:
                print("No observation standardization parameters found")
            
            print("RL policy loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading RL policy: {e}")
            import traceback
            traceback.print_exc()
            self.pretrained_policy = None
            return False

    def _get_environment_data(self):
        """Get depth images, proprioceptive data, and raycast targets from environment."""
        # Get depth images
        depth_images = self.env.get_depth_images()
        if depth_images is None:
            # If no depth images available, create dummy data
            depth_images = torch.zeros(self.env.num_envs, *self.depth_image_shape, device=self.device)
        else:
            # Take the most recent depth image if there's a buffer
            if len(depth_images.shape) == 4:  # (num_envs, buffer_len, height, width)
                depth_images = depth_images[:, -1]  # Take latest frame
            depth_images = depth_images.to(self.device)
            
        # Get proprioceptive data (base velocities)
        if hasattr(self.env, 'base_lin_vel') and hasattr(self.env, 'base_ang_vel'):
            proprio_data = torch.cat([
                self.env.base_lin_vel,  # [num_envs, 3]
                self.env.base_ang_vel   # [num_envs, 3]
            ], dim=-1)  # [num_envs, 6]
        else:
            # Dummy proprioceptive data if not available
            proprio_data = torch.zeros(self.env.num_envs, self.proprio_dim, device=self.device)

        # Get raycast targets (ground truth)
        if hasattr(self.env, '_get_raycast_distances'):
            raycast_targets = self.env._get_raycast_distances(normalize=False)
        else:
            # Dummy raycast data if not available
            raycast_targets = torch.zeros(self.env.num_envs, self.num_raycast_outputs, device=self.device)

        return depth_images, proprio_data, raycast_targets

    def _convert_raycast_distances_to_points(self, distances, env_ids=None):
        """Convert raycast distances to 3D world points for visualization.
        
        Args:
            distances: Raycast distances tensor [num_envs, num_rays] or [num_rays]
            env_ids: Environment IDs to visualize (default: all environments)
            
        Returns:
            List of 3D points for each environment
        """
        if not hasattr(self.env, 'ray_caster') or self.env.ray_caster is None:
            return []
            
        # Handle single environment case
        if len(distances.shape) == 1:
            distances = distances.unsqueeze(0)
            
        if env_ids is None:
            env_ids = range(min(distances.shape[0], self.env.num_envs))
        elif isinstance(env_ids, int):
            env_ids = [env_ids]

        all_points = []
        
        for i in env_ids:
            if i >= distances.shape[0]:
                continue
                
            # Get robot position and orientation
            base_pos = self.env.root_states[i, :3]
            base_quat = self.env.root_states[i, 3:7]
            
            # Get ray directions for this environment
            ray_directions = self.env.ray_caster.ray_directions[i, :]  # [num_rays, 3]
            
            # Apply robot orientation to ray directions
            from isaacgym.torch_utils import quat_rotate
            world_ray_directions = quat_rotate(base_quat.unsqueeze(0).repeat(ray_directions.shape[0], 1), 
                                             ray_directions)
            
            # Calculate ray origins (robot position + sensor offset)
            if hasattr(self.env.cfg, 'raycaster') and hasattr(self.env.cfg.raycaster, 'offset_pos'):
                sensor_offset = torch.tensor(self.env.cfg.raycaster.offset_pos, device=self.device)
                ray_origins = base_pos + quat_rotate(base_quat.unsqueeze(0), sensor_offset.unsqueeze(0))
            else:
                ray_origins = base_pos.unsqueeze(0)
            
            # Calculate hit points: origin + direction * distance
            env_distances = distances[i].unsqueeze(1)  # [num_rays, 1]
            hit_points = ray_origins + world_ray_directions * env_distances
            
            all_points.append(hit_points)
            
        return all_points

    def _visualize_estimator_predictions(self, iteration):
        """Visualize estimator predictions vs ground truth."""
        if not self.enable_visualization or not hasattr(self.env, 'draw_points'):
            return
            
        # Run estimator inference on current data
        with torch.no_grad():
            depth_images, proprio_data, raycast_targets = self._get_environment_data()
            
            # Get estimator predictions
            estimated_distances = self.estimator(depth_images, proprio_data)
            
            # Convert to 3D points for visualization (only visualize first few environments for performance)
            max_vis_envs = min(4, self.env.num_envs)
            env_ids = range(max_vis_envs)
            
            # Convert ground truth and predictions to 3D points
            gt_points = self._convert_raycast_distances_to_points(raycast_targets[:max_vis_envs], env_ids)
            pred_points = self._convert_raycast_distances_to_points(estimated_distances[:max_vis_envs], env_ids)
            
            # Draw points for each environment
            for i, env_id in enumerate(env_ids):
                if i < len(gt_points) and i < len(pred_points):
                    # Draw ground truth in red
                    self.env.draw_points(env_id, gt_points[i], color=(1, 0, 0), size=0.01)
                    # Draw predictions in blue
                    self.env.draw_points(env_id, pred_points[i], color=(0, 1, 1), size=0.02)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop for terrain estimator."""
        # Initialize writer if needed
        if hasattr(self, 'log_dir') and self.log_dir is not None and self.writer is None and not self.disable_logs:
            self._initialize_writer()

        # Randomize initial episode lengths if requested
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Training loop setup
        ep_infos = []
        loss_buffer = deque(maxlen=100)

        # Sync parameters for distributed training
        if hasattr(self, 'is_distributed') and self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Main training loop
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Turn off debug visualization because we are visualizing estimator predictions
        self.env.debug_viz = False
        for it in range(start_iter, tot_iter):
            start = time.time()

            # Data collection phase
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    # IMPORTANT: Get current environment data
                    depth_images, proprio_data, raycast_targets = self._get_environment_data()

                    # Store data for training (act method stores the data)
                    dummy_actions = self.alg.act(depth_images, proprio_data, raycast_targets)
                    # self.env.visualize_depth(0)
                    # Step environment with random actions (or pretrained policy actions)
                    if hasattr(self, 'pretrained_policy') and self.pretrained_policy is not None:
                        # Get observations for policy
                        if hasattr(self.env, 'obs_buf'):
                            # Old interface
                            observations = self.env.obs_buf
                        else:
                            # New interface
                            observations = self.env.get_observations()[0] if hasattr(self.env, 'get_observations') else None
                            
                        if observations is None:
                            print("Warning: Could not get observations for pretrained policy, using random actions")
                            actions = torch.randn(self.env.num_envs, self.env.num_actions, device=self.env.device) * 0.5
                        else:
                            # Apply observation normalization if available
                            if hasattr(self, 'obs_mean') and self.obs_mean is not None and hasattr(self, 'obs_var') and self.obs_var is not None:
                                normalized_obs = (observations - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
                                obs_for_policy = normalized_obs.to(self.device)
                            else:
                                obs_for_policy = observations.to(self.device)
                                
                            # Forward through policy
                            with torch.no_grad():
                                actions = self.pretrained_policy.act_inference(obs_for_policy)
                    else:
                        # Use random actions for data collection
                        actions = torch.randn(self.env.num_envs, self.env.num_actions, device=self.env.device) * 0.5

                    # Step environment
                    obs, priv_obs, rewards, dones, infos = self.env.step(actions)
                    
                    # Visualization
                    if self.enable_visualization and it % self.visualization_interval == 0:
                        self.env.vis.clear()
                        self._visualize_estimator_predictions(it)
                    
                    # Process environment step (rewards and infos are not used for estimator training)
                    self.alg.process_env_step(rewards, dones, infos)

                    # Logging - collect estimator-specific infos instead of episode infos
                    if hasattr(self, 'log_dir') and self.log_dir is not None:
                        # For terrain estimator, we don't need episode rewards/infos
                        # We'll log estimator metrics in the main training loop
                        pass

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Update estimator
            loss_dict = self.alg.update()
            loss_buffer.append(loss_dict["estimation"])

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Get additional training metrics
            training_infos = {
                "learning_rate": self.alg.optimizer.param_groups[0]['lr'],
                "grad_norm": getattr(self.alg, 'last_grad_norm', 0.0),
            }

            # Logging and saving
            if hasattr(self, 'log_dir') and self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"estimator_{it}.pt"))

            # No need to clear episode infos for estimator training
            # ep_infos.clear()

            # Save code state on first iteration
            if it == start_iter and not self.disable_logs:
                self._save_code_state()

        # Save final model
        if hasattr(self, 'log_dir') and self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"estimator_{self.current_learning_iteration}.pt"))

    def _initialize_writer(self):
        """Initialize logging writer."""
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _save_code_state(self):
        """Save code state for reproducibility."""
        if hasattr(self, 'log_dir') and self.log_dir and not self.disable_logs:
            git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
            if hasattr(self, 'logger_type') and self.logger_type in ["wandb", "neptune"] and git_file_paths:
                for path in git_file_paths:
                    self.writer.save_file(path)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        # -- Training metrics
        if "training_infos" in locs:
            for key, value in locs["training_infos"].items():
                self.writer.add_scalar(f"Train/{key}", value, locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["loss_buffer"]) > 0:
            self.writer.add_scalar("Train/mean_estimation_loss", statistics.mean(locs["loss_buffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        )

        # -- Losses
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""

        # -- Training metrics
        if "training_infos" in locs:
            for key, value in locs["training_infos"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.6f}\n"""

        if len(locs["loss_buffer"]) > 0:
            log_string += f"""{'Mean estimation loss:':>{pad}} {statistics.mean(locs['loss_buffer']):.4f}\n"""

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if hasattr(self, 'logger_type') and self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- load optimizer if used
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # -- load current learning iteration
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_estimator(self, device=None):
        """Get estimator for inference."""
        self.alg.policy.eval()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def play(self, num_episodes: int = 10, deterministic: bool = True, enable_visualization: bool = True):
        """Play/inference mode for terrain estimator visualization.
        
        Args:
            num_episodes: Number of episodes to run
            deterministic: Whether to use deterministic actions (if pretrained policy available)
            enable_visualization: Whether to enable visualization
        """
        print(f"Starting terrain estimator play mode for {num_episodes} episodes...")
        
        # Set estimator to evaluation mode
        self.estimator.eval()
        
        # Enable visualization
        old_viz_setting = getattr(self, 'enable_visualization', False)
        self.enable_visualization = enable_visualization
        
        # Reset environment
        self.env.reset()
        
        episode_count = 0
        step_count = 0
        
        with torch.no_grad():
            while episode_count < num_episodes:
                # Get current environment data
                depth_images, proprio_data, raycast_targets = self._get_environment_data()
                
                # Get estimator predictions
                estimated_distances = self.estimator(depth_images, proprio_data)
                
                # Calculate estimation error
                mse_error = torch.mean((estimated_distances - raycast_targets) ** 2)
                mae_error = torch.mean(torch.abs(estimated_distances - raycast_targets))
                
                # Print metrics every 100 steps
                if step_count % 100 == 0:
                    print(f"Step {step_count}: MSE={mse_error:.4f}, MAE={mae_error:.4f}")
                
                # Visualization
                if enable_visualization:
                    self.env.vis.clear()
                    self._visualize_estimator_predictions(step_count)
                    
                    # Optional: visualize depth images
                    # if hasattr(self.env, 'visualize_depth'):
                    #     self.env.visualize_depth(0, f"Depth - Step {step_count}")
                
                # Get actions for environment stepping
                if hasattr(self, 'pretrained_policy') and self.pretrained_policy is not None:
                    # Use pretrained policy
                    if hasattr(self.env, 'obs_buf'):
                        observations = self.env.obs_buf
                    else:
                        observations = self.env.get_observations()[0] if hasattr(self.env, 'get_observations') else None
                        
                    if observations is not None:
                        # Apply observation normalization if available
                        if hasattr(self, 'obs_mean') and self.obs_mean is not None and hasattr(self, 'obs_var') and self.obs_var is not None:
                            normalized_obs = (observations - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
                            obs_for_policy = normalized_obs.to(self.device)
                        else:
                            obs_for_policy = observations.to(self.device)
                            
                        if deterministic:
                            actions = self.pretrained_policy.act_inference(obs_for_policy)
                        else:
                            actions = self.pretrained_policy.act(obs_for_policy)
                    else:
                        actions = torch.randn(self.env.num_envs, self.env.num_actions, device=self.env.device) * 0.1
                else:
                    # Use small random actions
                    actions = torch.randn(self.env.num_envs, self.env.num_actions, device=self.env.device) * 0.1
                
                # Step environment
                obs, priv_obs, rewards, dones, infos = self.env.step(actions)
                
                # Check for episode completion
                if torch.any(dones):
                    completed_episodes = torch.sum(dones).item()
                    episode_count += completed_episodes
                    print(f"Completed {completed_episodes} episodes. Total: {episode_count}/{num_episodes}")
                
                step_count += 1
                
                # Add small delay for visualization
                if enable_visualization:
                    import time
                    time.sleep(0.02)
        
        # Restore visualization setting
        self.enable_visualization = old_viz_setting
        
        print(f"Terrain estimator play completed after {step_count} steps and {episode_count} episodes.")


