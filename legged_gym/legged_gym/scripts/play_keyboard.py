from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import select
import termios
import tty

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math_utils import quat_apply_yaw

import numpy as np
import torch
import time


def set_raw_noecho(fd):
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new)
    return old


def get_key(timeout=0.01):
    rlist, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return ''


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── Single-robot keyboard-control overrides ──
    env_cfg.env.num_envs = 1
    env_cfg.env.max_episode_length = 99999     # disable timeout reset
    env_cfg.env.episode_length_s = 99999.0
    env_cfg.commands.resampling_time = 99999   # disable automatic command resampling
    env_cfg.commands.curriculum = False
    env_cfg.init_state.randomize_rot = True
    env_cfg.init_state.rot_randomization_range = [1.5708, 1.5708]  # +90° yaw
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.terrain_length = 16.0
    env_cfg.terrain.terrain_width = 16.0
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # print(f"Keyboard control ready.  w/s: x vel ±1.0  a/d: y vel ±1.0  q/e: yaw ±0.5")
    # print(f"Terrain: col 0 = pillar_field, col 1 = sin_curve_channel")
    # print(f"Robot spawned at pillar_field centre.")

    fd_stdin = sys.stdin.fileno()
    old_termios = set_raw_noecho(fd_stdin)
    try:
        while True:
            step_start = time.time()

            key = get_key()
            lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
            if key == 'w':
                lin_vel_x = 1.0
            elif key == 's':
                lin_vel_x = -1.0
            elif key == 'a':
                lin_vel_y = 1.0
            elif key == 'd':
                lin_vel_y = -1.0
            elif key == 'q':
                ang_vel_yaw = 1.0
            elif key == 'e':
                ang_vel_yaw = -1.0

            env.commands[:] = torch.tensor([[lin_vel_x, lin_vel_y, ang_vel_yaw, 0.0]],
                                           device=env.device)

            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            if CAMERA_FOLLOW:
                robot_pos = env.root_states[0, :3]
                robot_quat = env.base_quat[0:1]
                behind = quat_apply_yaw(robot_quat, torch.tensor([[-3.0, 0.0, 2.0]],
                                         device=env.device))
                forward = quat_apply_yaw(robot_quat, torch.tensor([[1.0, 0.0, 0.0]],
                                         device=env.device))
                env.set_camera((robot_pos + behind[0]).cpu().numpy(),
                               (robot_pos + forward[0]).cpu().numpy())

            if REALTIME_MODE:
                step_duration = time.time() - step_start
                sleep_time = env.dt - step_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        termios.tcsetattr(fd_stdin, termios.TCSADRAIN, old_termios)


if __name__ == '__main__':
    REALTIME_MODE = True
    CAMERA_FOLLOW = False  # True=跟随机器人, False=固定视角
    args = get_args()
    play(args)
