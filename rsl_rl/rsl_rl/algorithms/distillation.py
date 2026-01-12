# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: Union[StudentTeacher, StudentTeacherRecurrent]
    """The student teacher model."""

    def __init__(
        self,
        policy: Union[StudentTeacher, StudentTeacherRecurrent],
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: Optional[dict] = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs):
        # compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, dones in self.storage.generator():

                # inference the student for gradient computation
                actions = self.policy.act_inference(obs)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel


class EstimatorDistillation:
    """Distillation algorithm for training a terrain estimator to predict raycast distances from depth images."""

    def __init__(
        self,
        estimator,  # TerrainEstimator model
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: Optional[dict] = None,
        **kwargs  # Additional parameters for the estimator
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # estimator components
        self.policy = estimator  # Using 'policy' name for compatibility with runner
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = EstimatorTransition()
        self.last_hidden_states = None

        # training parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        elif loss_type == "l1":
            self.loss_fn = nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber, l1")

        self.num_updates = 0
        self.last_grad_norm = 0.0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, depth_shape, proprio_shape, raycast_shape
    ):
        # create rollout storage for estimator training
        self.storage = EstimatorRolloutStorage(
            num_envs,
            num_transitions_per_env,
            depth_shape,
            proprio_shape,
            raycast_shape,
            self.device,
        )

    def act(self, depth_images, proprio_data, raycast_targets):
        """
        Store data for training. Unlike standard RL, we don't need to 'act' but just collect data.
        """
        # Store the current observations and targets
        self.transition.depth_images = depth_images
        self.transition.proprio_data = proprio_data
        self.transition.raycast_targets = raycast_targets
        
        # For compatibility, return a dummy action (not used in estimator training)
        return torch.zeros(depth_images.shape[0], 1, device=self.device)

    def process_env_step(self, rewards, dones, infos):
        # record the transition (rewards and dones not used for estimator training)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_estimation_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            for depth_images, proprio_data, raycast_targets, dones in self.storage.generator():
                # Forward pass through estimator
                raycast_predictions = self.policy.act_inference(depth_images, proprio_data)

                # Estimation loss (regression loss)
                estimation_loss = self.loss_fn(raycast_predictions, raycast_targets)

                # total loss
                loss = loss + estimation_loss
                mean_estimation_loss += estimation_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    
                    # Calculate gradient norm before clipping
                    grad_norm = 0.0
                    for param in self.policy.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    self.last_grad_norm = grad_norm
                    
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_estimation_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"estimation": mean_estimation_loss}

        return loss_dict

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel


class EstimatorTransition:
    """Data structure for storing estimator training transitions."""
    
    def __init__(self):
        self.depth_images = None
        self.proprio_data = None
        self.raycast_targets = None
    
    def clear(self):
        self.depth_images = None
        self.proprio_data = None
        self.raycast_targets = None


class EstimatorRolloutStorage:
    """Rollout storage for estimator training data."""
    
    def __init__(self, num_envs, num_transitions_per_env, depth_shape, proprio_shape, raycast_shape, device):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        
        # Storage buffers
        self.depth_images = torch.zeros(
            num_transitions_per_env, num_envs, *depth_shape, device=device
        )
        self.proprio_data = torch.zeros(
            num_transitions_per_env, num_envs, proprio_shape[0], device=device
        )
        self.raycast_targets = torch.zeros(
            num_transitions_per_env, num_envs, raycast_shape[0], device=device
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        
        self.step = 0
    
    def add_transitions(self, transition):
        """Add a transition to the storage."""
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        self.depth_images[self.step].copy_(transition.depth_images)
        self.proprio_data[self.step].copy_(transition.proprio_data)
        self.raycast_targets[self.step].copy_(transition.raycast_targets)
        
        self.step += 1
    
    def clear(self):
        """Clear the storage for next rollout."""
        self.step = 0
    
    def generator(self, batch_size=None):
        """Generate batches for training."""
        if batch_size is None:
            batch_size = self.num_envs
        
        # Simple generator that yields all data (can be made more sophisticated)
        for start_idx in range(0, self.num_transitions_per_env * self.num_envs, batch_size):
            end_idx = min(start_idx + batch_size, self.num_transitions_per_env * self.num_envs)
            
            # Flatten the stored data
            depth_flat = self.depth_images.view(-1, *self.depth_images.shape[2:])
            proprio_flat = self.proprio_data.view(-1, self.proprio_data.shape[2])
            raycast_flat = self.raycast_targets.view(-1, self.raycast_targets.shape[2])
            dones_flat = self.dones.view(-1, 1)
            
            yield (
                depth_flat[start_idx:end_idx],
                proprio_flat[start_idx:end_idx],
                raycast_flat[start_idx:end_idx],
                dones_flat[start_idx:end_idx]
            )
