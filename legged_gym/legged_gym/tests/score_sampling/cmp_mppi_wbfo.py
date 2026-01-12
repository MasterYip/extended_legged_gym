import os
from random import sample
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from legged_gym.envs.batch_rollout.robot_traj_grad_sampling import RobotTrajGradSampling
from legged_gym.envs.batch_rollout.robot_traj_grad_sampling_config import RobotTrajGradSamplingCfg
from traj_sampling.optimizer import create_wbfo_optimizer, create_avwbfo_optimizer
from traj_sampling.spline import SplineBase
import torch


class OptimizationComparison:
    def __init__(self, device='cuda:0'):
        """Initialize comparison environment with both MPPI and WBFO.

        Args:
            device: Device to run computations on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.spline = SplineBase(device=self.device)

        # Noise parameters
        self.noise_std = 2.0 # (Change it in below)

        # MPPI parameters
        self.action_size = 2  # For 2D trajectories
        self.num_samples = 100  #(Change it in below) Number of trajectories to sample
        self.temp_sample = 0.1  # Temperature for weighting samples
        self.horizon_samples = 64  # Number of trajectory samples
        self.horizon_nodes = 8  # Number of control nodes

        # WBFO parameters

        class WBFOCfg:
            class env:
                num_actions = 2

            class trajectory_opt:
                # Sampling parameters
                num_samples = 100  #(Change it in below) Number of samples to generate at each diffusion step (Nsample)
                temp_sample = 0.1
                horizon_samples = 64  # Horizon length in samples (Hsample)
                horizon_nodes = 8  # Number of control nodes within horizon (Hnode)
                # NOTE: for AVWBFO
                gamma = 1.00  # Discount factor for rewards (For testing system, )

            sim_device = device

        self.wbfo_cfg = WBFOCfg()

        self.wbfo_optimizer = create_wbfo_optimizer(self.wbfo_cfg)

        self.avwbfo_optimizer = create_avwbfo_optimizer(self.wbfo_cfg)

        print(f"Optimization comparison initialized on device: {self.device}")

    def define_test_landscapes(self):
        """Define test cost landscapes for comparison."""
        landscapes = {}

        # 1. Simple quadratic cost - modified to return per-step rewards
        def quadratic_cost(trajectories):
            """Simple quadratic cost: penalize distance from target.
            Returns reward for each timestep in trajectory.
            """
            target = torch.tensor([5.0, 5.0], device=self.device)
            # Calculate distance from target at each step
            # Shape: [batch_size, timesteps]
            dist = torch.norm(trajectories[:, :, -1, :] - target, dim=2)
            # Return negative distance as reward (higher is better)
            return -dist

        # 2. Obstacle avoidance cost - modified to return per-step rewards
        def obstacle_cost(trajectories):
            """Penalize proximity to obstacles while reaching target.
            Returns reward for each timestep in trajectory.
            """
            target = torch.tensor([8.0, 8.0], device=self.device)
            obstacles = [
                (3.0, 3.0, 1.5),  # x, y, radius
                (6.0, 6.0, 1.0),
                (2.0, 7.0, 1.2)
            ]

            batch_size = trajectories.shape[0]
            timesteps = trajectories.shape[1]

            # Target reaching reward - per step
            # Shape: [batch_size, timesteps]
            dist_to_target = torch.norm(trajectories[:, :, -1, :] - target, dim=2)
            target_reward = -dist_to_target

            # Obstacle avoidance penalty - per step
            # Shape: [batch_size, timesteps]
            obstacle_penalty = torch.zeros((batch_size, timesteps), device=self.device)

            for obs_x, obs_y, obs_radius in obstacles:
                obs_pos = torch.tensor([obs_x, obs_y], device=self.device)
                # Distance to obstacle at each step
                # Shape: [batch_size, timesteps]
                dist_to_obs = torch.norm(trajectories[:, :, -1, :] - obs_pos, dim=2)
                # Penalty increases as we get closer than radius
                # Shape: [batch_size, timesteps]
                penalty = torch.clamp(obs_radius - dist_to_obs, min=0.0)
                # Add penalty for this obstacle
                obstacle_penalty += penalty * 5.0

            # Return combined reward (higher is better)
            return target_reward - obstacle_penalty

        # 3. Dynamic cost with moving target - modified to return per-step rewards
        def dynamic_cost(trajectories):
            """Cost with a moving target.
            Returns reward for each timestep in trajectory.
            """
            batch_size = trajectories.shape[0]
            time_steps = trajectories.shape[1]

            # Target moves in a circle
            target_traj = torch.zeros(time_steps, 2, device=self.device)
            t = torch.linspace(0, 2*np.pi, time_steps, device=self.device)
            target_traj[:, 0] = 5.0 + 2.0 * torch.cos(t)
            target_traj[:, 1] = 5.0 + 2.0 * torch.sin(t)

            # Expanded target trajectory for batch calculation
            # Shape: [batch_size, timesteps, 2]
            expanded_target = target_traj.unsqueeze(0).expand(batch_size, -1, -1)

            # Distance between trajectory endpoint and moving target at each timestep
            # Shape: [batch_size, timesteps]
            dist = torch.norm(trajectories[:, :, -1, :] - expanded_target, dim=2)

            # Return negative distance as reward (higher is better)
            return -dist

        landscapes["quadratic"] = quadratic_cost
        landscapes["obstacle"] = obstacle_cost
        landscapes["dynamic"] = dynamic_cost

        return landscapes

    def generate_initial_trajectory(self):
        """Generate an initial trajectory for optimization."""
        # Linear trajectory from starting point
        start_point = torch.tensor([0.0, 0.0], device=self.device)
        initial_direction = torch.tensor([0.5, 0.5], device=self.device)

        # Create nodes for a straight-line trajectory
        node_trajectories = torch.zeros(
            (self.horizon_nodes + 1, self.action_size),
            device=self.device
        )

        for i in range(self.horizon_nodes + 1):
            t = i / self.horizon_nodes
            node_trajectories[i] = start_point + t * initial_direction * 10.0

        return node_trajectories

    def node2dense(self, nodes):
        """Convert trajectory nodes to dense trajectory using spline interpolation."""
        return self.wbfo_optimizer.node2dense(nodes)

    def sample_trajectory(self, mean_traj):
        """Sample a trajectory around the mean trajectory.

        Args:
            mean_traj: Mean trajectory [horizon_nodes+1, action_dim]

        Returns:
            Sampled trajectories [num_samples, horizon_samples, action_dim]
        """
        eps = torch.randn(
            (self.num_samples, self.horizon_nodes + 1, self.action_size),
            device=self.device
        ) * self.noise_std

        return mean_traj + eps

    def mppi_optimization(self, mean_traj, cost_function, num_iterations=10):
        """Perform MPPI optimization on a trajectory.

        Args:
            mean_traj: Initial trajectory [horizon_nodes+1, action_dim]
            cost_function: Function to evaluate trajectory cost
            num_iterations: Number of optimization iterations

        Returns:
            Optimized trajectory and optimization history
        """
        # Initialize history tracking
        history = {
            'mean_traj': [mean_traj.clone()],
            'costs': [],
            'best_cost': float('-inf'),
            'best_traj': mean_traj.clone(),
            'time': []
        }

        # Current best trajectory
        current_traj = mean_traj.clone()

        start_time = time.time()
        for iteration in range(num_iterations):
            iter_start = time.time()

            # Convert to dense trajectories for evaluation
            samples = self.sample_trajectory(current_traj)
            dense_samples = self.node2dense(samples)
            dense_samples_with_time = dense_samples.unsqueeze(2)

            # Shape: [batch_size, timesteps]
            step_rewards = cost_function(dense_samples_with_time)
            costs = torch.sum(step_rewards, dim=1)

            # Normalize costs for numerical stability
            cost_mean = costs.mean()
            cost_std = costs.std() + 1e-6
            normalized_costs = (costs - cost_mean) / cost_std

            # Compute weights using softmax
            weights = torch.softmax(normalized_costs / self.temp_sample, dim=0)

            # Update trajectory using weighted average
            current_traj = torch.sum(weights.view(-1, 1, 1) * samples, dim=0)

            # Evaluate current trajectory
            dense_current = self.node2dense(current_traj.unsqueeze(0))
            dense_current_with_time = dense_current.unsqueeze(2)

            # Evaluate per-step rewards for current trajectory
            current_step_rewards = cost_function(dense_current_with_time)

            # Sum for total reward
            current_cost = torch.sum(current_step_rewards).item()

            # Track history
            history['mean_traj'].append(current_traj.clone())
            history['costs'].append(current_cost)
            history['time'].append(time.time() - iter_start)

            # Update best trajectory if needed
            if current_cost > history['best_cost']:
                history['best_cost'] = current_cost
                history['best_traj'] = current_traj.clone()

            print(f"MPPI Iteration {iteration+1}/{num_iterations}, Cost: {current_cost:.4f}")

        total_time = time.time() - start_time
        print(f"MPPI Optimization completed in {total_time:.2f}s")

        return history['best_traj'], history

    def wbfo_optimization(self, mean_traj, cost_function, num_iterations=10):
        """Perform WBFO optimization on a trajectory.

        Args:
            mean_traj: Initial trajectory [horizon_nodes+1, action_dim]
            cost_function: Function to evaluate trajectory cost
            num_iterations: Number of optimization iterations

        Returns:
            Optimized trajectory and optimization history
        """
        # Initialize history tracking
        history = {
            'mean_traj': [mean_traj.clone()],
            'costs': [],
            'best_cost': float('-inf'),
            'best_traj': mean_traj.clone(),
            'time': []
        }

        # Current best trajectory
        current_traj = mean_traj.clone()

        start_time = time.time()
        for iteration in range(num_iterations):
            iter_start = time.time()

            # Convert to dense trajectories for evaluation
            samples = self.sample_trajectory(current_traj)
            dense_samples = self.node2dense(samples)
            dense_samples_with_time = dense_samples.unsqueeze(2)

            # Evaluate costs - get per-step rewards
            step_rewards = cost_function(dense_samples_with_time)

            # WBFO can directly use per-step rewards
            current_traj = self.wbfo_optimizer.optimize(
                current_traj,
                samples,
                step_rewards
            ).squeeze(0)

            # Evaluate current trajectory
            dense_current = self.node2dense(current_traj.unsqueeze(0))
            dense_current_with_time = dense_current.unsqueeze(2)

            # Get per-step rewards
            current_step_rewards = cost_function(dense_current_with_time)

            # Sum for total reward (for consistent comparison with MPPI)
            current_cost = torch.sum(current_step_rewards).item()

            # Track history
            history['mean_traj'].append(current_traj.clone())
            history['costs'].append(current_cost)
            history['time'].append(time.time() - iter_start)

            # Update best trajectory if needed
            if current_cost > history['best_cost']:
                history['best_cost'] = current_cost
                history['best_traj'] = current_traj.clone()

            print(f"WBFO Iteration {iteration+1}/{num_iterations}, Cost: {current_cost:.4f}")

        total_time = time.time() - start_time
        print(f"WBFO Optimization completed in {total_time:.2f}s")

        return history['best_traj'], history

    def avwbfo_optimization(self, mean_traj, cost_function, num_iterations=10):
        """Perform AVWBFO optimization on a trajectory.

        Args:
            mean_traj: Initial trajectory [horizon_nodes+1, action_dim]
            cost_function: Function to evaluate trajectory cost
            num_iterations: Number of optimization iterations

        Returns:
            Optimized trajectory and optimization history
        """
        # Initialize history tracking
        history = {
            'mean_traj': [mean_traj.clone()],
            'costs': [],
            'best_cost': float('-inf'),
            'best_traj': mean_traj.clone(),
            'time': []
        }

        # Current best trajectory
        current_traj = mean_traj.clone()

        start_time = time.time()
        for iteration in range(num_iterations):
            iter_start = time.time()

            # Convert to dense trajectories for evaluation
            samples = self.sample_trajectory(current_traj)
            dense_samples = self.node2dense(samples)
            dense_samples_with_time = dense_samples.unsqueeze(2)

            # Evaluate costs - get per-step rewards
            step_rewards = cost_function(dense_samples_with_time)

            # AVWBFO can directly use per-step rewards
            current_traj = self.avwbfo_optimizer.optimize(
                current_traj,
                samples,
                step_rewards
            ).squeeze(0)

            # Evaluate current trajectory
            dense_current = self.node2dense(current_traj.unsqueeze(0))
            dense_current_with_time = dense_current.unsqueeze(2)

            # Get per-step rewards
            current_step_rewards = cost_function(dense_current_with_time)

            # Sum for total reward (for consistent comparison with MPPI)
            current_cost = torch.sum(current_step_rewards).item()

            # Track history
            history['mean_traj'].append(current_traj.clone())
            history['costs'].append(current_cost)
            history['time'].append(time.time() - iter_start)

            # Update best trajectory if needed
            if current_cost > history['best_cost']:
                history['best_cost'] = current_cost
                history['best_traj'] = current_traj.clone()

            print(f"AVWBFO Iteration {iteration+1}/{num_iterations}, Cost: {current_cost:.4f}")

        total_time = time.time() - start_time
        print(f"AVWBFO Optimization completed in {total_time:.2f}s")

        return history['best_traj'], history

    def visualize_comparison(self, landscape_name, mppi_history, wbfo_history, cost_function):
        """Visualize the comparison between MPPI and WBFO optimization.

        Args:
            landscape_name: Name of the cost landscape used
            mppi_history: History from MPPI optimization
            wbfo_history: History from WBFO optimization
            cost_function: Function to evaluate trajectories
        """
        # Create a figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)

        # 1. Plot cost histories
        ax_cost = fig.add_subplot(gs[0, 0])
        ax_cost.plot(mppi_history['costs'], 'r-', label='MPPI')
        ax_cost.plot(wbfo_history['costs'], 'b-', label='WBFO')
        ax_cost.set_xlabel('Iteration')
        ax_cost.set_ylabel('Reward (negative cost)')
        ax_cost.set_title(f'Optimization Progress - {landscape_name}')
        ax_cost.legend()
        ax_cost.grid(True)

        # 2. Plot computation time per iteration
        ax_time = fig.add_subplot(gs[0, 1])
        ax_time.plot(mppi_history['time'], 'r-', label='MPPI')
        ax_time.plot(wbfo_history['time'], 'b-', label='WBFO')
        ax_time.set_xlabel('Iteration')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_title('Computation Time per Iteration')
        ax_time.legend()
        ax_time.grid(True)

        # 3. Plot cumulative time
        ax_cum_time = fig.add_subplot(gs[0, 2])
        mppi_cum_time = np.cumsum(mppi_history['time'])
        wbfo_cum_time = np.cumsum(wbfo_history['time'])
        ax_cum_time.plot(mppi_cum_time, mppi_history['costs'], 'r-', label='MPPI')
        ax_cum_time.plot(wbfo_cum_time, wbfo_history['costs'], 'b-', label='WBFO')
        ax_cum_time.set_xlabel('Cumulative Time (s)')
        ax_cum_time.set_ylabel('Reward')
        ax_cum_time.set_title('Reward vs. Computation Time')
        ax_cum_time.legend()
        ax_cum_time.grid(True)

        # 4. Plot trajectories in 2D space
        ax_traj = fig.add_subplot(gs[1, :])

        # Get best trajectories
        mppi_best_traj = mppi_history['best_traj']
        wbfo_best_traj = wbfo_history['best_traj']

        # Convert to dense trajectories for visualization
        mppi_dense = self.node2dense(mppi_best_traj.unsqueeze(0)).squeeze(0)
        wbfo_dense = self.node2dense(wbfo_best_traj.unsqueeze(0)).squeeze(0)

        # Plot trajectories
        ax_traj.plot(mppi_dense[:, 0].cpu().numpy(), mppi_dense[:, 1].cpu().numpy(),
                     'r-', linewidth=2, label=f'MPPI Best (R={mppi_history["best_cost"]:.2f})')
        ax_traj.plot(wbfo_dense[:, 0].cpu().numpy(), wbfo_dense[:, 1].cpu().numpy(),
                     'b-', linewidth=2, label=f'WBFO Best (R={wbfo_history["best_cost"]:.2f})')

        # Plot initial trajectory
        initial_traj = mppi_history['mean_traj'][0]
        initial_dense = self.node2dense(initial_traj.unsqueeze(0)).squeeze(0)
        ax_traj.plot(initial_dense[:, 0].cpu().numpy(), initial_dense[:, 1].cpu().numpy(),
                     'k--', linewidth=1, label='Initial')

        # Plot start and end points
        ax_traj.plot(initial_dense[0, 0].cpu().numpy(), initial_dense[0, 1].cpu().numpy(), 'ko', markersize=8, label='Start')

        # Plot control points for better visualization
        mppi_nodes = mppi_best_traj.cpu().numpy()
        wbfo_nodes = wbfo_best_traj.cpu().numpy()
        ax_traj.plot(mppi_nodes[:, 0], mppi_nodes[:, 1], 'ro', markersize=4)
        ax_traj.plot(wbfo_nodes[:, 0], wbfo_nodes[:, 1], 'bo', markersize=4)

        # If we have obstacles in the landscape, visualize them
        if landscape_name == 'obstacle':
            obstacles = [
                (3.0, 3.0, 1.5),  # x, y, radius
                (6.0, 6.0, 1.0),
                (2.0, 7.0, 1.2)
            ]

            for obs_x, obs_y, obs_radius in obstacles:
                circle = plt.Circle((obs_x, obs_y), obs_radius, fill=True, alpha=0.3, color='gray')
                ax_traj.add_patch(circle)

            # Target
            target = (8.0, 8.0)
            circle_target = plt.Circle(target, 0.3, fill=True, color='green')
            ax_traj.add_patch(circle_target)
            ax_traj.annotate("Target", xy=target, xytext=(target[0], target[1]+0.5))

        elif landscape_name == 'dynamic':
            # Visualize the moving target trajectory
            t = torch.linspace(0, 2*np.pi, self.horizon_samples+1, device=self.device)
            target_x = 5.0 + 2.0 * torch.cos(t)
            target_y = 5.0 + 2.0 * torch.sin(t)
            ax_traj.plot(target_x.cpu().numpy(), target_y.cpu().numpy(), 'g-', label='Target Path')

        elif landscape_name == 'quadratic':
            # Visualize the target point
            target = (5.0, 5.0)
            circle_target = plt.Circle(target, 0.3, fill=True, color='green')
            ax_traj.add_patch(circle_target)
            ax_traj.annotate("Target", xy=target, xytext=(target[0], target[1]+0.5))

        ax_traj.set_xlabel('X Position')
        ax_traj.set_ylabel('Y Position')
        ax_traj.set_title(f'Optimized Trajectories - {landscape_name}')
        ax_traj.legend(loc='upper left')
        ax_traj.grid(True)
        ax_traj.set_aspect('equal')

        # Show plot
        plt.tight_layout()
        plt.savefig(f'comparison_{landscape_name}.png')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare MPPI and WBFO optimization methods")
    parser.add_argument('--landscape', type=str, default='all', choices=['quadratic', 'obstacle', 'dynamic', 'all'],
                        help='Cost landscape to use for comparison')
    parser.add_argument('--iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples per iteration')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')
    parser.add_argument('--noise', type=float, default=0.5, help='Noise standard deviation for sampling')

    args = parser.parse_args()

    # Initialize comparison object
    comp = OptimizationComparison(device=args.device)
    comp.num_samples = args.samples
    comp.noise_std = args.noise

    # Get test landscapes
    landscapes = comp.define_test_landscapes()

    # Determine which landscapes to test
    test_landscapes = list(landscapes.keys()) if args.landscape == 'all' else [args.landscape]

    # Run comparisons
    for landscape_name in test_landscapes:
        print(f"\n=== Testing on {landscape_name} landscape ===")

        # Generate initial trajectory
        initial_traj = comp.generate_initial_trajectory()

        # Optimize using MPPI
        print("\nOptimizing with MPPI...")
        mppi_best, mppi_history = comp.mppi_optimization(
            initial_traj, landscapes[landscape_name], num_iterations=args.iterations
        )

        # # Optimize using WBFO
        # print("\nOptimizing with WBFO...")
        # wbfo_best, wbfo_history = comp.wbfo_optimization(
        #     initial_traj, landscapes[landscape_name], num_iterations=args.iterations
        # )

        # Optimize using AVWBFO
        print("\nOptimizing with AVWBFO...")
        avwbfo_best, avwbfo_history = comp.avwbfo_optimization(
            initial_traj, landscapes[landscape_name], num_iterations=args.iterations
        )

        # Visualize comparison
        comp.visualize_comparison(landscape_name, mppi_history, avwbfo_history, landscapes[landscape_name])


if __name__ == "__main__":
    main()
