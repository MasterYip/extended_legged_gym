# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import select
import sys
import termios
import tty

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path

import numpy as np
import torch
import time
import isaacgym.gymapi as gymapi


class KeyboardInput:
    def __init__(self):
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.old_settings)

    def read_keys(self):
        if self.fd is None:
            return []

        keys = []
        while True:
            rlist, _, _ = select.select([self.fd], [], [], 0.0)
            if not rlist:
                break
            key = os.read(self.fd, 1).decode(errors="ignore")
            if key:
                keys.append(key.lower())
        return keys


CAMERA_MODES = ("rear_side", "rear_top")
CAMERA_MODE_NAMES = {
    "rear_side": "侧后方",
    "rear_top": "后上方",
}


def _update_follow_camera(env, camera_state, robot_index=0, mode="rear_side"):
    """Keep the viewer camera near the robot without following body jitter."""
    if getattr(env, "viewer", None) is None:
        return camera_state

    base_pos = env.root_states[robot_index, :3].detach().cpu().numpy()
    base_quat = env.root_states[robot_index, 3:7].detach().cpu().numpy()
    x, y, z, w = base_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if camera_state is None:
        camera_state = {
            "yaw": yaw,
            "base_z": base_pos[2],
            "position": None,
            "lookat": None,
        }
    else:
        yaw_delta = np.arctan2(np.sin(yaw - camera_state["yaw"]), np.cos(yaw - camera_state["yaw"]))
        camera_state["yaw"] += 0.08 * yaw_delta
        camera_state["base_z"] += 0.03 * (base_pos[2] - camera_state["base_z"])

    stable_base_pos = base_pos.copy()
    stable_base_pos[2] = camera_state["base_z"]
    yaw = camera_state["yaw"]

    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)

    if mode == "rear_top":
        target_position = stable_base_pos - 1.25 * forward + np.array([0.0, 0.0, 1.8], dtype=np.float64)
        target_lookat = stable_base_pos + 0.2 * forward + np.array([0.0, 0.0, 0.15], dtype=np.float64)
    else:
        target_position = (
            stable_base_pos
            - 1.6 * forward
            + 0.75 * left
            + np.array([0.0, 0.0, 0.65], dtype=np.float64)
        )
        target_lookat = stable_base_pos + 0.25 * forward + np.array([0.0, 0.0, 0.2], dtype=np.float64)

    if camera_state["position"] is None:
        camera_state["position"] = target_position
        camera_state["lookat"] = target_lookat
    else:
        camera_state["position"] += 0.12 * (target_position - camera_state["position"])
        camera_state["lookat"] += 0.12 * (target_lookat - camera_state["lookat"])

    env.set_camera(camera_state["position"], camera_state["lookat"])
    return camera_state


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # env.dubug.viz = True
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = min(env_cfg.terrain.num_rows, 5)
    env_cfg.terrain.num_cols = min(env_cfg.terrain.num_cols, 5)
    env_cfg.terrain.curriculum = env_cfg.terrain.curriculum
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "resampling_time"):
        env_cfg.commands.resampling_time = 1e9

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    lighting_cfg = getattr(env_cfg, "lighting", None)
    l_color_cfg = getattr(lighting_cfg, "base_light_color", [0.5, 0.5, 0.5])
    l_ambient_cfg = getattr(lighting_cfg, "base_light_ambient", [0.1, 0.1, 0.1])
    l_direction_cfg = getattr(lighting_cfg, "base_light_direction", [0.0, 0.0, 1.0])
    l_color = gymapi.Vec3(l_color_cfg[0], l_color_cfg[1], l_color_cfg[2])
    l_ambient = gymapi.Vec3(l_ambient_cfg[0], l_ambient_cfg[1], l_ambient_cfg[2])
    l_direction = gymapi.Vec3(l_direction_cfg[0], l_direction_cfg[1], l_direction_cfg[2])
    env.gym.set_light_parameters(env.sim, 0, l_color, l_ambient, l_direction)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.policy, path)
        print('Exported policy as jit script to: ', path)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    try:
        load_path = get_load_path(
            log_root,
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        load_run_dir = os.path.dirname(load_path)
    except Exception:
        load_run_dir = log_root

    logger = Logger(env.dt, output_root=load_run_dir)
    robot_index = 0 # which robot is used for logging
    joint_index = 0 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    camera_mode_idx = 0
    camera_state = None
    if FOLLOW_CAMERA:
        camera_state = _update_follow_camera(env, camera_state, robot_index, CAMERA_MODES[camera_mode_idx])

    # Realtime management variables
    realtime_factor_window = []
    realtime_factor_window_size = 50
    last_print_time = time.time()
    print_interval = 2.0  # Print realtime factor every 2 seconds
    
    if REALTIME_MODE:
        print(f"Running in realtime mode (target dt={env.dt:.4f}s)")
    else:
        print("Running at maximum speed (no realtime constraints)")

    lin_vel_x = 0.0  # 前进/后退速度
    lin_vel_y = 0.0  # 侧移速度
    ang_vel_yaw = 0.0  # 偏航角速度
    heading = 0.0  # 预留

    camera_key_text = ", c=toggle camera side-rear/rear-top" if FOLLOW_CAMERA else ""
    print(
        f"Keyboard: wasd=linear velocity, qe=yaw velocity, z=reset command{camera_key_text}."
    )

    keyboard = KeyboardInput()
    keyboard.__enter__()
    try:
        if keyboard.fd is None:
            print("stdin is not a TTY; keyboard control is disabled.")

        for i in range(int(env.max_episode_length*10)):

            step_start_time = time.time()

            # 键盘控制指令（wasdqe控制）
            keys = keyboard.read_keys()
            lin_step = 0.1
            yaw_step = 0.1

            for key in keys:
                if key == 'w':
                    lin_vel_x += lin_step
                elif key == 's':
                    lin_vel_x -= lin_step
                elif key == 'a':
                    lin_vel_y += lin_step
                elif key == 'd':
                    lin_vel_y -= lin_step
                elif key == 'q':
                    ang_vel_yaw += yaw_step
                elif key == 'e':
                    ang_vel_yaw -= yaw_step
                elif key == 'z':
                    lin_vel_x, lin_vel_y, ang_vel_yaw, heading = 0.0, 0.0, 0.0, 0.0  # 重置
                elif key == 'c' and FOLLOW_CAMERA:
                    camera_mode_idx = (camera_mode_idx + 1) % len(CAMERA_MODES)
                    if camera_state is not None:
                        camera_state["position"] = None
                    print(f"\n摄像头视角: {CAMERA_MODE_NAMES[CAMERA_MODES[camera_mode_idx]]}")

            # 限制范围
            lin_vel_x = np.clip(lin_vel_x, -5.0, 5.0)
            lin_vel_y = np.clip(lin_vel_y, -1.5, 1.5)
            ang_vel_yaw = np.clip(ang_vel_yaw, -2.0, 2.0)
            heading = np.clip(heading, -1.0, 1.0)
            command = [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
            command = (command + [0.0] * env.commands.shape[1])[:env.commands.shape[1]]
            env.commands[:] = torch.tensor(command, dtype=env.commands.dtype, device=env.commands.device)
            if hasattr(env, "compute_observations"):
                env.compute_observations()
                obs = env.get_observations()
            print(
                f"当前命令: lin_vel_x={lin_vel_x:.2f}, lin_vel_y={lin_vel_y:.2f}, "
                f"ang_vel_yaw={ang_vel_yaw:.2f}, heading={heading:.2f}, "
                f"{'camera=' + CAMERA_MODE_NAMES[CAMERA_MODES[camera_mode_idx]] if FOLLOW_CAMERA else ''}",
                end="\r",
            )

            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            if FOLLOW_CAMERA:
                camera_state = _update_follow_camera(env, camera_state, robot_index, CAMERA_MODES[camera_mode_idx])

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA and not FOLLOW_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if ENABLE_LOGGING:
                if i < stop_state_log:
                    logger.log_states(
                        {
                            'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale +  env.default_dof_pos[robot_index, joint_index].item(),
                            'dof_pos': env.dof_pos[robot_index, joint_index] .item(),
                            'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                            'dof_vel_1': env.dof_vel[robot_index, 1].item(),
                            'dof_vel_2': env.dof_vel[robot_index, 2].item(),
                            'dof_torque': env.torques[robot_index, joint_index].item(),
                            'dof_torque_1': env.torques[robot_index, 1].item(),
                            'dof_torque_2': env.torques[robot_index, 2].item(),
                            'command_x': env.commands[robot_index, 0].item(),
                            'command_y': env.commands[robot_index, 1].item(),
                            'command_yaw': env.commands[robot_index, 2].item(),
                            'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                            'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                            'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                            'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                            'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                            }
                        )
                elif i==stop_state_log:
                    logger.plot_states()
                if  0 < i < stop_rew_log:
                    if infos["episode"]:
                        num_episodes = torch.sum(env.reset_buf).item()
                        if num_episodes>0:
                            logger.log_rewards(infos["episode"], num_episodes)
                elif i==stop_rew_log:
                    logger.print_rewards()

            # Realtime management
            if REALTIME_MODE:
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                    
                # Calculate realtime factor
                realtime_factor = env.dt / step_duration if step_duration > 0 else float('inf')
                realtime_factor = max(0.0, min(realtime_factor, 1.0))  # Clamp between 0 and 1
                realtime_factor_window.append(realtime_factor)
                    
                # Maintain window size
                if len(realtime_factor_window) > realtime_factor_window_size:
                    realtime_factor_window.pop(0)
                    
                # Print realtime factor periodically (disabled)
                # current_time = time.time()
                # if current_time - last_print_time >= print_interval:
                #     avg_realtime_factor = np.mean(realtime_factor_window)
                #     min_realtime_factor = np.min(realtime_factor_window)
                #     max_realtime_factor = np.max(realtime_factor_window)
                #     print(f"Step {i}: Realtime factor: {avg_realtime_factor:.2f}x "
                #           f"(min: {min_realtime_factor:.2f}x, max: {max_realtime_factor:.2f}x)")
                #     last_print_time = current_time
                    
                # Sleep to maintain realtime if computation was faster than env.dt
                sleep_time = env.dt - step_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
            # If not realtime mode, run as fast as possible (no sleep)
    finally:
        keyboard.__exit__(None, None, None)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FOLLOW_CAMERA = False
    ENABLE_LOGGING = True
    REALTIME_MODE = True  # Set to False to run at maximum speed
    args = get_args()
    play(args)
