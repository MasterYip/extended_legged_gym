#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch

from legged_gym.envs.elspider_air.batch_rollout.elspider_air_traj_grad_sampling import ElSpiderAirTrajGradSampling
from legged_gym.envs.elspider_air.batch_rollout.elspider_air_traj_grad_sampling_config import ElSpiderAirTrajGradSamplingCfg, ElSpiderAirTrajGradSamplingCfgPPO
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_args

from copy import deepcopy

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_vis', action='store_true', default=True,
                        help='Enable debug visualization')
    parser.add_argument('--num_envs', type=int, default=4,
                        help='Number of environments to create')
    parser.add_argument('--rollout_envs', type=int, default=64,
                        help='Number of rollout environments per main environment')
    parser.add_argument('--horizon_nodes', type=int, default=5,
                        help='Number of control nodes in the horizon')
    parser.add_argument('--horizon_samples', type=int, default=20,
                        help='Number of samples in the horizon')
    parser.add_argument('--num_diffuse_steps', type=int, default=10,
                        help='Number of diffusion steps')
    parser.add_argument('--num_steps', type=int, default=300,
                        help='Number of simulation steps')
    parser.add_argument('--optimize_interval', type=int, default=1,
                        help='Number of steps between trajectory optimizations')
    parser.add_argument('--show_contacts', action='store_true', default=False,
                        help='Visualize foot contacts')
    parser.add_argument('--command', type=str, default='walk_forward',
                        choices=['walk_forward', 'walk_backward', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right'],
                        help='Command to send to the robot')
    parser.add_argument('--robot', type=str, default='elspider_air',
                        choices=['elspider_air', 'elspider_air_plan', 'cassie', 'anymal_c', 'go2'],
                        help='Robot type to use')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode (no GUI)')
    args = parser.parse_args()
    return args


def get_command(command_name):
    """Get command vector based on command name."""
    commands = {
        'walk_forward': [1.0, 0.0, 0.0, 0.0],
        'walk_backward': [-1.0, 0.0, 0.0, 0.0],
        'strafe_left': [0.0, 0.5, 0.0, 0.0],
        'strafe_right': [0.0, -1.0, 0.0, 0.0],
        'turn_left': [0.0, 0.0, 0.5, 0.0],
        'turn_right': [0.0, 0.0, -0.5, 0.0],
    }
    return commands.get(command_name, [1.0, 0.0, 0.0, 0.0])


def main():
    """Test the trajectory gradient sampling environment."""
    args = get_args()

    args_ext = parse_arguments()

    # Set the task based on the selected robot
    if args_ext.robot == 'cassie':
        args_ext.task = "cassie_traj_grad_sampling"
    elif args_ext.robot == 'anymal_c':
        args_ext.task = "anymal_c_traj_grad_sampling"
    elif args_ext.robot == 'go2':
        args_ext.task = "go2_traj_grad_sampling"
    elif args_ext.robot == 'elspider_air_plan':
        args_ext.task = "elspider_air_plan_grad_sampling"
    else:
        args_ext.task = "elspider_air_traj_grad_sampling"

    args.headless = args_ext.headless

    # Import required classes based on robot selection
    if args_ext.robot == 'cassie':
        from legged_gym.envs.cassie.cassie_traj_grad_sampling import CassieTrajGradSampling
        from legged_gym.envs.cassie.cassie_traj_grad_sampling_config import CassieTrajGradSamplingCfg, CassieTrajGradSamplingCfgPPO

        # Register the CassieTrajGradSampling environment and config with the task registry
        task_registry.register("cassie_traj_grad_sampling",
                               CassieTrajGradSampling,
                               CassieTrajGradSamplingCfg,
                               CassieTrajGradSamplingCfgPPO)
    elif args_ext.robot == 'anymal_c':
        from legged_gym.envs.anymal_c.batch_rollout.anymal_c_traj_grad_sampling import AnymalCTrajGradSampling
        from legged_gym.envs.anymal_c.batch_rollout.anymal_c_traj_grad_sampling_config import AnymalCTrajGradSamplingCfg, AnymalCTrajGradSamplingCfgPPO

        # Register the AnymalCTrajGradSampling environment and config with the task registry
        task_registry.register("anymal_c_traj_grad_sampling",
                               AnymalCTrajGradSampling,
                               AnymalCTrajGradSamplingCfg,
                               AnymalCTrajGradSamplingCfgPPO)
    elif args_ext.robot == 'go2':
        from legged_gym.envs.go2.batch_rollout.go2_traj_grad_sampling import Go2TrajGradSampling
        from legged_gym.envs.go2.batch_rollout.go2_traj_grad_sampling_config import Go2TrajGradSamplingCfg, Go2TrajGradSamplingCfgPPO

        # Register the Go2TrajGradSampling environment and config with the task registry
        task_registry.register("go2_traj_grad_sampling",
                               Go2TrajGradSampling,
                               Go2TrajGradSamplingCfg,
                               Go2TrajGradSamplingCfgPPO)
    else:
        from legged_gym.envs.elspider_air.batch_rollout.elspider_air_traj_grad_sampling import ElSpiderAirTrajGradSampling
        from legged_gym.envs.elspider_air.batch_rollout.elspider_air_traj_grad_sampling_config import ElSpiderAirTrajGradSamplingCfg, ElSpiderAirTrajGradSamplingCfgPPO

        # Register the ElSpiderAirTrajGradSampling environment and config with the task registry
        task_registry.register("elspider_air_traj_grad_sampling",
                               ElSpiderAirTrajGradSampling,
                               ElSpiderAirTrajGradSamplingCfg,
                               ElSpiderAirTrajGradSamplingCfgPPO)

    # Get the environment config from the task registry
    env_cfg, _ = task_registry.get_cfgs(args_ext.task)

    # prepare environment
    env, _ = task_registry.make_env(name=args_ext.task, args=args, env_cfg=env_cfg)

    # Reset the environment to get initial observations
    obs = env.reset()

    # Get the command for the robot
    command = get_command(args_ext.command)

    # Set the command for all environments
    commands = torch.tensor(command, device=env.device).unsqueeze(0).repeat(env.total_num_envs, 1)
    env.set_all_commands(commands)
    env._draw_debug_vis()

    print(f"Command: {command}")
    print(f"Starting simulation for {args_ext.num_steps} steps...")

    # Track rewards and actions for replay
    rewards_history = []
    run_time_history = []
    recorded_actions = []  # Store actions for replay

    # Track individual reward components
    last_reward_sums = deepcopy(env.episode_sums)
    episode_reward_sums = {}
    reward_component_history = {}

    # Start recording actions
    if hasattr(env, 'start_recording'):
        env.start_recording()
        print("Started recording actions")

    # Run the simulation
    start_time = time.time()
    # DEBUG: use_rl_policy
    use_rl_policy = False
    try:
        for i in range(args_ext.num_steps):
            if not use_rl_policy:
                # Optimize trajectories every few steps to improve performance
                if i % args_ext.optimize_interval == 0:
                    t0 = time.time()
                    print(f"\nStep {i}: Optimizing trajectories...")

                    # Use batch optimization for all environments
                    results = env.optimize_all_trajectories(initial=(i == 0))

                    # Record optimization time
                    opt_time = time.time() - t0
                    print(f"Optimization completed in {opt_time:.4f} seconds")
                    run_time_history.append(opt_time)

                # Get first action from optimized trajectory for each environment
                actions = torch.stack([env.action_trajectories[j, 0] for j in range(env.num_envs)])
            else:
                obs = env.get_observations()
                actions = env.traj_grad_sampler.rl_policy.act_inference(obs)

            # Record actions for replay
            recorded_actions.append(actions.clone())

            # Step the environment
            obs, _, rewards, dones, infos = env.step(actions)

            # Record rewards
            rewards_history.append(rewards[0].item())
            # rewards_history.append(env.rew_buf[0].item()*50)  # Use rew_buf for current rewards

            # Track individual reward components from environment episode sums
            if hasattr(env, 'episode_sums') and not use_rl_policy:
                total = 0
                for reward_name, reward_sum in env.episode_sums.items():
                    if reward_name not in reward_component_history:
                        reward_component_history[reward_name] = []
                    # Log the mean reward component value across all environments
                    reward_component_history[reward_name].append((reward_sum[0] - last_reward_sums[reward_name][0]).cpu()) # mean traj reward of env1
                    total += reward_sum[0] - last_reward_sums[reward_name][0]
                if "total_reward" not in reward_component_history:  # Skip total_reward
                    reward_component_history["total_reward"] = []
                reward_component_history["total_reward"].append(total.cpu())
                last_reward_sums = deepcopy(env.episode_sums)

            # Process episode info for reward component logging (similar to on_policy_runner)
            # NOTE: Due to non-reset, episode sum is always empty
            if "episode" in infos:
                ep_info = infos["episode"]
                # Extract reward components from episode info
                for key, value in ep_info.items():
                    if key.startswith('rew_'):
                        reward_name = key[4:]  # Remove 'rew_' prefix
                        if reward_name not in episode_reward_sums:
                            episode_reward_sums[reward_name] = []
                        if isinstance(value, torch.Tensor):
                            episode_reward_sums[reward_name].append(value.item())
                        else:
                            episode_reward_sums[reward_name].append(value)

            print(f"Step {i}: Actions: {actions}")
            print(f"Step {i}: Reward = {rewards.mean().item():.3f}")

            # Print reward components every 50 steps
            if i % 50 == 0 and len(reward_component_history) > 0:
                print("Reward components:")
                for reward_name, values in reward_component_history.items():
                    if len(values) > 0:
                        print(f"  {reward_name}: {values[-1]:.4f}")

            # Print progress and average reward every 10 steps
            if i % 10 == 0:
                print(f"Step {i}/{args_ext.num_steps}: Total Reward = {rewards.mean().item():.3f}")

            # Reset environments that are done
            if dones.any():
                env.reset_idx(torch.nonzero(dones).squeeze(-1))

                # Re-apply commands
                for j in range(env.num_envs):
                    if dones[j]:
                        env.commands[j, :] = torch.tensor(command, device=env.device)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    total_time = time.time() - start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds")
    print(f"Average reward: {np.mean(rewards_history):.3f}")
    print(f"Average optimization time: {np.mean(run_time_history):.4f} seconds")

    # Print final reward component summary
    if len(reward_component_history) > 0:
        print("\nFinal reward component averages:")
        for reward_name, values in reward_component_history.items():
            if len(values) > 0:
                print(f"  {reward_name}: {np.mean(values):.4f}")

    if len(episode_reward_sums) > 0:
        print("\nEpisode reward component averages:")
        for reward_name, values in episode_reward_sums.items():
            if len(values) > 0:
                print(f"  {reward_name}: {np.mean(values):.4f}")

    # Stop recording
    if hasattr(env, 'stop_recording'):
        env.stop_recording()
        print("Stopped recording actions")

    # Plot the rewards
    plt.figure(figsize=(15, 10))

    # Plot total rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Total Rewards During Trajectory Optimization')
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.grid(True)

    # Plot reward components
    if len(reward_component_history) > 0:
        plt.subplot(2, 2, 2)
        for reward_name, values in reward_component_history.items():
            if len(values) > 0:
                plt.plot(values, label=reward_name)
        plt.title('Reward Components During Trajectory Optimization')
        plt.xlabel('Step')
        plt.ylabel('Reward Component Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

    # Plot optimization times
    plt.subplot(2, 2, 3)
    plt.plot(run_time_history)
    plt.title('Trajectory Optimization Times')
    plt.xlabel('Optimization Iteration')
    plt.ylabel('Time (seconds)')
    plt.grid(True)

    # Plot episode reward sums if available
    if len(episode_reward_sums) > 0:
        plt.subplot(2, 2, 4)
        for reward_name, values in episode_reward_sums.items():
            if len(values) > 0:
                plt.plot(values, label=reward_name)
        plt.title('Episode Reward Components')
        plt.xlabel('Episode')
        plt.ylabel('Reward Component Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('trajectory_optimization_rewards.png', dpi=150, bbox_inches='tight')
    print(f"Reward plot saved to trajectory_optimization_rewards.png")

    # Stack all recorded actions for easy replay
    all_actions = torch.stack(recorded_actions) if recorded_actions else None

    # Replay option - check if the environment has replay capability
    if hasattr(env, 'replay_blocking'):
        print("\nStarting replay of recorded actions...")
        while True:
            try:
                env.replay_blocking()
            except KeyboardInterrupt:
                print("Replay interrupted by user")
                return
    elif all_actions is not None:
        # Manual replay if the environment doesn't have built-in replay
        print("\nStarting manual replay of recorded actions...")
        try:
            # Reset environment for replay
            obs = env.reset()

            # Re-apply commands for consistency
            commands = torch.tensor(command, device=env.device).unsqueeze(0).repeat(env.total_num_envs, 1)
            env.set_all_commands(commands)

            # Replay the recorded actions
            for i, action in enumerate(all_actions):
                obs, _, rewards, dones, _ = env.step(action)
                if i % 10 == 0:
                    print(f"Replay step {i}/{len(all_actions)}: Reward = {rewards.mean().item():.3f}")
        except KeyboardInterrupt:
            print("Replay interrupted by user")
    else:
        print("\nNo actions recorded for replay")


if __name__ == "__main__":
    main()
