# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from typing import Optional, Union
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state

# Global variable to control interface version support
# Set to True to use old legged_gym interface (rsl_rl 1.0.2 style)
# Set to False to use new interface (rsl_rl 3.3.0+ style)
USE_OLD_INTERFACE = None  # Auto-detect by default


def set_old_interface(use_old: bool = True):
    """Set the interface version to use.

    Args:
        use_old: If True, use old legged_gym interface. If False, use new interface.
    """
    global USE_OLD_INTERFACE
    USE_OLD_INTERFACE = use_old


def detect_interface_version(env: VecEnv) -> bool:
    """Automatically detect which interface version the environment uses.

    Args:
        env: The environment instance

    Returns:
        True if old interface is detected, False if new interface
    """
    # Check for old interface signatures
    has_old_attributes = (
        hasattr(env, 'num_privileged_obs') and
        hasattr(env, 'obs_buf') and
        hasattr(env, 'privileged_obs_buf') and
        hasattr(env, 'rew_buf') and
        hasattr(env, 'reset_buf')
    )

    # Check for new interface method
    has_new_method = hasattr(env, 'get_observations') and callable(env.get_observations)

    # Old interface: has old attributes but no new method or new method returns simple tensor
    if has_old_attributes:
        if not has_new_method:
            return True
        else:
            # Test if get_observations returns old-style single tensor or new-style tuple
            try:
                result = env.get_observations()
                return not isinstance(result, tuple)
            except:
                return True

    return False


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: Optional[str] = None, device="cpu"):
        self.device = device
        self.env = env
        self.log_dir = log_dir  # Store log_dir early

        # Automatically detect interface version if not explicitly set
        global USE_OLD_INTERFACE
        if USE_OLD_INTERFACE is None:
            USE_OLD_INTERFACE = detect_interface_version(env)

        self.use_old_interface = USE_OLD_INTERFACE

        # Parse configuration based on interface version
        self._parse_config(train_cfg)

        # Configure multi-GPU if available
        self._configure_multi_gpu()

        # Setup observations and training type
        num_obs, num_privileged_obs = self._setup_observations()

        # Initialize policy
        policy = self._initialize_policy(num_obs, num_privileged_obs)

        # Setup RND if configured
        self._setup_rnd()

        # Setup symmetry if configured
        self._setup_symmetry()

        # Initialize algorithm
        self.alg = self._initialize_algorithm(policy)

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Setup empirical normalization
        self._setup_normalization(num_obs, num_privileged_obs)

        # Initialize storage
        self._initialize_storage(num_obs, num_privileged_obs)

        # Setup logging
        self._setup_logging()

        # Initialize environment if using old interface
        if self.use_old_interface:
            self._initialize_old_interface()

        if self.training_type == "distillation":
            teacher_model_path = self.cfg.get("teacher_model_path", None)
            if teacher_model_path and os.path.exists(teacher_model_path):
                self.load(teacher_model_path)
                print(f"Loaded teacher model from {teacher_model_path}")
            else:
                raise ValueError(
                    "Teacher model path not specified or does not exist. "
                    "Please provide a valid path to the teacher model."
                )

    def _parse_config(self, train_cfg: dict):
        """Parse training configuration based on interface version."""
        if self.use_old_interface:
            # Old interface: train_cfg has "runner", "algorithm", "policy" keys
            self.cfg = train_cfg["runner"]
            self.alg_cfg = train_cfg["algorithm"]
            self.policy_cfg = train_cfg["policy"]
        else:
            # New interface: train_cfg is the direct config or has different structure
            if "runner" in train_cfg:
                self.cfg = train_cfg["runner"]
                self.alg_cfg = train_cfg["algorithm"]
                self.policy_cfg = train_cfg["policy"]
            else:
                self.cfg = train_cfg
                self.alg_cfg = train_cfg["algorithm"]
                self.policy_cfg = train_cfg["policy"]

    def _setup_observations(self) -> tuple[int, int]:
        """Setup observation dimensions based on interface version."""
        if self.use_old_interface:
            # Old interface: use direct attributes
            num_obs = self.env.num_obs
            num_privileged_obs = getattr(self.env, 'num_privileged_obs', num_obs)
            if num_privileged_obs is None:
                num_privileged_obs = num_obs
            # Enhanced distillation support for old interface
            # Check if this is distillation training
            algorithm_name = self.cfg["algorithm_class_name"]
            if algorithm_name == "Distillation":
                self.training_type = "distillation"
                # For distillation, privileged obs should be teacher observations
                self.privileged_obs_type = "teacher" if num_privileged_obs != num_obs else None
            else:
                self.training_type = "rl"
                self.privileged_obs_type = "critic" if num_privileged_obs != num_obs else None

        else:
            # New interface: resolve training type and observations
            if "class_name" in self.alg_cfg:
                if self.alg_cfg["class_name"] == "PPO":
                    self.training_type = "rl"
                elif self.alg_cfg["class_name"] == "Distillation":
                    self.training_type = "distillation"
                else:
                    raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")
            else:
                self.training_type = "rl"  # Default for old config style

            # Get observation dimensions from environment
            obs, extras = self.env.get_observations()
            num_obs = obs.shape[1]

            # Resolve privileged observations
            if self.training_type == "rl":
                if "critic" in extras["observations"]:
                    self.privileged_obs_type = "critic"
                    num_privileged_obs = extras["observations"]["critic"].shape[1]
                else:
                    self.privileged_obs_type = None
                    num_privileged_obs = num_obs
            elif self.training_type == "distillation":
                if "teacher" in extras["observations"]:
                    self.privileged_obs_type = "teacher"
                    num_privileged_obs = extras["observations"]["teacher"].shape[1]
                else:
                    self.privileged_obs_type = None
                    num_privileged_obs = num_obs
            else:
                self.privileged_obs_type = None
                num_privileged_obs = num_obs

        return num_obs, num_privileged_obs

    def _initialize_policy(self, num_obs: int, num_privileged_obs: int):
        """Initialize policy based on configuration."""
        if self.use_old_interface:
            # Old interface: use policy_class_name
            policy_class = eval(self.cfg["policy_class_name"])
        else:
            # New interface: use class_name from policy config
            if "class_name" in self.policy_cfg:
                policy_class = eval(self.policy_cfg.pop("class_name"))
            else:
                # Fallback for old config style
                policy_class = eval(self.cfg["policy_class_name"])

        policy = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        return policy

    def _setup_rnd(self):
        """Setup Random Network Distillation if configured."""
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            if not self.use_old_interface:
                # Only available in new interface
                _, extras = self.env.get_observations()
                rnd_state = extras["observations"].get("rnd_state")
                if rnd_state is None:
                    raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
                # Get dimension of rnd gated state
                num_rnd_state = rnd_state.shape[1]
                # Add rnd gated state to config
                self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
                # Scale down the rnd weight with timestep
                self.alg_cfg["rnd_cfg"]["weight"] *= self.env.unwrapped.step_dt

    def _setup_symmetry(self):
        """Setup symmetry if configured."""
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # This is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = self.env

    def _initialize_algorithm(self, policy):
        """Initialize the algorithm."""
        if self.use_old_interface:
            # Old interface: use algorithm_class_name
            alg_class = eval(self.cfg["algorithm_class_name"])
        else:
            # New interface: use class_name from algorithm config
            if "class_name" in self.alg_cfg:
                alg_class = eval(self.alg_cfg.pop("class_name"))
            else:
                # Fallback for old config style
                alg_class = eval(self.cfg["algorithm_class_name"])

        # Create algorithm instance
        alg_kwargs = dict(self.alg_cfg)
        if hasattr(self, 'multi_gpu_cfg') and self.multi_gpu_cfg:
            alg_kwargs["multi_gpu_cfg"] = self.multi_gpu_cfg

        return alg_class(policy, device=self.device, **alg_kwargs)

    def _setup_normalization(self, num_obs: int, num_privileged_obs: int):
        """Setup empirical normalization if enabled."""
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

    def _initialize_storage(self, num_obs: int, num_privileged_obs: int):
        """Initialize rollout storage."""
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

    def _setup_logging(self):
        """Setup logging configuration."""
        self.disable_logs = hasattr(self, 'is_distributed') and self.is_distributed and self.gpu_global_rank != 0
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _initialize_old_interface(self):
        """Initialize environment for old interface compatibility."""
        # Old interface environments expect a reset call during initialization
        if hasattr(self.env, 'reset'):
            # Check if reset expects env_ids argument (old interface style)
            try:
                # Try calling reset without arguments first
                self.env.reset()
            except TypeError:
                # If that fails, try with empty env_ids
                try:
                    self.env.reset([])
                except:
                    # Some environments might need different initialization
                    pass

    def _get_observations(self):
        """Get observations based on interface version."""
        if self.use_old_interface:
            # Old interface: use direct buffer access or method calls
            obs = self.env.get_observations()
            privileged_obs = self.env.get_privileged_observations()
            privileged_obs = privileged_obs if privileged_obs is not None else obs
            return obs, privileged_obs
        else:
            # New interface: use get_observations method
            obs, extras = self.env.get_observations()
            privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
            return obs, privileged_obs

    def _process_step_info(self, infos):
        """Process step information based on interface version."""
        if self.use_old_interface:
            # Old interface: infos might have different structure
            return infos
        else:
            # New interface: infos have standard structure
            return infos

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop."""
        # Initialize writer if needed
        if hasattr(self, 'log_dir') and self.log_dir is not None and self.writer is None and not self.disable_logs:
            self._initialize_writer()

        # Check teacher loading for distillation
        if hasattr(self, 'training_type') and self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # Randomize initial episode lengths if requested
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs, privileged_obs = self._get_observations()
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        # Training loop setup
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # RND buffers if available
        if hasattr(self.alg, 'rnd') and self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Sync parameters for distributed training
        if hasattr(self, 'is_distributed') and self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Main training loop
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout phase
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs)

                    # Step environment
                    if self.use_old_interface:
                        # Old interface: step returns (obs, privileged_obs, rewards, dones, infos)
                        step_result = self.env.step(actions.to(self.env.device))
                        if len(step_result) == 5:
                            obs, privileged_obs, rewards, dones, infos = step_result
                            if privileged_obs is None:
                                privileged_obs = obs
                        else:
                            # Some old interfaces might return 4 values
                            obs, rewards, dones, infos = step_result
                            privileged_obs = self.env.get_privileged_observations()
                            privileged_obs = privileged_obs if privileged_obs is not None else obs
                    else:
                        # New interface: step returns (obs, rewards, dones, infos)
                        obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                        if self.privileged_obs_type is not None:
                            privileged_obs = infos["observations"][self.privileged_obs_type]
                        else:
                            privileged_obs = obs

                    # Move to device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    privileged_obs = privileged_obs.to(self.device)

                    # Apply normalization
                    obs = self.obs_normalizer(obs)
                    privileged_obs = self.privileged_obs_normalizer(privileged_obs)

                    # Process environment step
                    processed_infos = self._process_step_info(infos)
                    self.alg.process_env_step(rewards, dones, processed_infos)

                    # Logging and bookkeeping
                    if hasattr(self, 'log_dir') and self.log_dir is not None:
                        self._update_episode_tracking(
                            rewards, dones, infos, ep_infos, rewbuffer, lenbuffer,
                            cur_reward_sum, cur_episode_length
                        )

                        # RND tracking if available
                        if hasattr(self.alg, 'rnd') and self.alg.rnd:
                            self._update_rnd_tracking(
                                rewards, dones, erewbuffer, irewbuffer,
                                cur_ereward_sum, cur_ireward_sum
                            )

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns for RL training
                if hasattr(self, 'training_type') and self.training_type == "rl":
                    # Clone the privileged_obs to get a normal tensor that can be used in autograd
                    privileged_obs_for_returns = privileged_obs.clone()
                    self.alg.compute_returns(privileged_obs_for_returns)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging and saving
            if hasattr(self, 'log_dir') and self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Update reward scales if multi-stage rewards
            if self.cfg.get("multi_stage_rewards", False) and len(rewbuffer) > 0 and\
                    self.env.update_reward_scales(statistics.mean(rewbuffer)):
                print("Updated reward scales to ", self.env.reward_scales_stage)
                rewbuffer.clear()
                cur_reward_sum *= 0.0

            # Clear episode infos
            ep_infos.clear()

            # Save code state on first iteration
            if it == start_iter and not self.disable_logs:
                self._save_code_state()

        # Save final model
        if hasattr(self, 'log_dir') and self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _initialize_writer(self):
        """Initialize logging writer."""
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            if hasattr(self.env, 'cfg'):
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            if hasattr(self.env, 'cfg'):
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _update_episode_tracking(self, rewards, dones, infos, ep_infos, rewbuffer, lenbuffer,
                                 cur_reward_sum, cur_episode_length):
        """Update episode tracking for logging."""
        if "episode" in infos:
            ep_infos.append(infos["episode"])
        elif "log" in infos:
            ep_infos.append(infos["log"])

        cur_reward_sum += rewards
        cur_episode_length += 1

        # Clear data for completed episodes
        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

    def _update_rnd_tracking(self, rewards, dones, erewbuffer, irewbuffer,
                             cur_ereward_sum, cur_ireward_sum):
        """Update RND reward tracking."""
        intrinsic_rewards = getattr(self.alg, 'intrinsic_rewards', None)
        if intrinsic_rewards is not None:
            cur_ereward_sum += rewards
            cur_ireward_sum += intrinsic_rewards

            new_ids = (dones > 0).nonzero(as_tuple=False)
            if len(new_ids) > 0:
                erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                cur_ereward_sum[new_ids] = 0
                cur_ireward_sum[new_ids] = 0

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

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
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
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if not USE_OLD_INTERFACE and self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)

            def policy(x): return self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

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
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _setup_distillation_training(self):
        """Setup distillation training components for old interface."""
        # Import distillation algorithm
        try:
            from rsl_rl.algorithms import Distillation
        except ImportError:
            raise ImportError("Distillation algorithm not found. Please implement it in rsl_rl.algorithms")

        # Load teacher model if specified
        teacher_model_path = getattr(self.cfg, 'teacher_model_path', None)
        if teacher_model_path and os.path.exists(teacher_model_path):
            teacher_checkpoint = torch.load(teacher_model_path, map_location=self.device)
            print(f"Loaded teacher model from {teacher_model_path}")
        else:
            teacher_checkpoint = None
            print("No teacher model specified or file not found")

        # Setup distillation algorithm
        self.alg = Distillation(
            self.alg.actor_critic,
            teacher_checkpoint=teacher_checkpoint,
            device=self.device,
            **self.alg_cfg
        )

        print("Distillation training setup completed")
