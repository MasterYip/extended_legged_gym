# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive playback for the external-condition ``el4090_envelop_2`` task."""

import os
import select
import sys
import termios
import time
import tty
import unicodedata

import isaacgym  # noqa: F401 -- Isaac Gym must be imported before torch-backed env modules.
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 -- registers tasks.
from legged_gym.utils import export_policy_as_jit, get_args, task_registry


TASK_NAME = "el4090_envelop_2"
EXPECTED_OBS_DIM = 83
ENVELOPE_GEOMETRY_NAMES = (
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
)
MORPHOLOGY_PRIOR_NAMES = (
    "morphology_front_prior",
    "morphology_middle_prior",
    "morphology_back_prior",
)
CAMERA_MODES = ("rear_side", "rear_top")
CAMERA_MODE_NAMES = {
    "rear_side": "侧后方",
    "rear_top": "后上方",
}
MORPHOLOGY_CYCLE_HALF_DURATION_S = 3.0


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
            readable, _, _ = select.select([self.fd], [], [], 0.0)
            if not readable:
                break
            key = os.read(self.fd, 1).decode(errors="ignore")
            if key:
                keys.append(key.lower())
        return keys


def _print_keyboard_table(follow_camera):
    rows = [
        ("W / S", "设置前进 / 后退速度"),
        ("A / D", "设置左移 / 右移速度"),
        ("Q / E", "设置左转 / 右转角速度"),
        ("X", "速度指令全部归零"),
        ("M", "切换到最大包络（默认初始状态）"),
        ("R", "随机采样一个包络"),
        ("F", "前腿 mammal 倾向测试包络"),
        ("O", "全昆虫型 → 全哺乳型 → 全昆虫型"),
    ]
    if follow_camera:
        rows.append(("C", "切换侧后方 / 后上方相机"))
    rows.append(("ESC", "退出"))

    def display_width(text):
        return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)

    def pad(text, width):
        return text + " " * max(0, width - display_width(text))

    key_width = max(display_width(key) for key, _ in rows + [("按键", "")])
    function_width = max(display_width(description) for _, description in rows + [("", "功能")])
    border = f"+{'-' * (key_width + 2)}+{'-' * (function_width + 2)}+"
    print("\n按键功能表")
    print(border)
    print(f"| {pad('按键', key_width)} | {pad('功能', function_width)} |")
    print(border)
    for key, description in rows:
        print(f"| {pad(key, key_width)} | {pad(description, function_width)} |")
    print(border)


def _condition_template(env):
    state = env.envelope_state
    return (0.5 * (state.low + state.high)).unsqueeze(0).repeat(env.num_envs, 1)


def _maximum_envelope_condition(env):
    """Build the largest physical footprint; priors are derived by the env API."""
    condition = _condition_template(env)
    state = env.envelope_state
    names = list(state.condition_names)
    for name in ("front_width", "middle_width", "back_width", "forward_limit"):
        index = names.index(name)
        condition[:, index] = state.high[index]
    backward_index = names.index("backward_limit")
    condition[:, backward_index] = state.low[backward_index]
    return condition


def _random_envelope_condition(env):
    condition = _condition_template(env)
    state = env.envelope_state
    names = list(state.condition_names)
    for name in ENVELOPE_GEOMETRY_NAMES:
        index = names.index(name)
        condition[:, index] = state.low[index] + torch.rand(
            env.num_envs,
            dtype=condition.dtype,
            device=condition.device,
        ) * (state.high[index] - state.low[index])
    return condition


def _front_mammal_condition(env):
    condition = _condition_template(env)
    state = env.envelope_state
    names = list(state.condition_names)
    selections = {
        "front_width": state.low,
        "forward_limit": state.high,
        "middle_width": state.high,
        "back_width": state.high,
        "backward_limit": state.high,
    }
    for name, bound in selections.items():
        index = names.index(name)
        condition[:, index] = bound[index]
    return condition


def _all_insect_condition(env):
    """Build an envelope whose derived morphology priors are all zero."""
    condition = _condition_template(env)
    state = env.envelope_state
    names = list(state.condition_names)
    selections = {
        "front_width": state.high,
        "middle_width": state.high,
        "back_width": state.high,
        "forward_limit": state.low,
        "backward_limit": state.high,
    }
    for name, bound in selections.items():
        index = names.index(name)
        condition[:, index] = bound[index]
    return condition


def _all_mammal_condition(env):
    """Build an envelope whose derived morphology priors are all one."""
    condition = _condition_template(env)
    state = env.envelope_state
    names = list(state.condition_names)
    selections = {
        "front_width": state.low,
        "middle_width": state.low,
        "back_width": state.low,
        "forward_limit": state.high,
        "backward_limit": state.low,
    }
    for name, bound in selections.items():
        index = names.index(name)
        condition[:, index] = bound[index]
    return condition


def _morphology_cycle_condition(insect, mammal, step, half_steps):
    """Cosine-interpolate insect -> mammal -> insect over two half cycles."""
    cycle_progress = min(max(float(step) / (2.0 * half_steps), 0.0), 1.0)
    mammal_blend = 0.5 - 0.5 * np.cos(2.0 * np.pi * cycle_progress)
    return torch.lerp(insect, mammal, float(mammal_blend))


def _transition_condition(start, target, step, transition_steps):
    progress = min(max(float(step) / transition_steps, 0.0), 1.0)
    blend = 0.5 - 0.5 * np.cos(np.pi * progress)
    return torch.lerp(start, target, float(blend))


def _new_morphology_cycle(env, *, diverse=False):
    insect = _all_insect_condition(env)
    return {
        "step": 0,
        "half_steps": max(1, int(round(MORPHOLOGY_CYCLE_HALF_DURATION_S / env.dt))),
        "insect": insect,
        "mammal": _all_mammal_condition(env),
        "diverse": diverse,
        "start": insect,
        "target": _random_envelope_condition(env) if diverse else None,
        "target_index": 1,
    }


def _apply_envelope(env, condition, *, reset_transition=True):
    if not hasattr(env, "set_envelope_condition"):
        raise AttributeError(
            "play_envelop_2.py requires the envelop_2 set_envelope_condition() API"
        )
    updated = env.set_envelope_condition(condition, derive_priors=True)
    if reset_transition and hasattr(env, "embedded_state_transition_time"):
        env.embedded_state_transition_time[:] = 0.0
    return updated


def _print_envelope_state(env, label, robot_index=0):
    condition = env._get_structure_condition()[robot_index]
    names = list(env.condition_names)
    geometry = "  ".join(
        f"{name}={condition[names.index(name)].item():.3f}"
        for name in ENVELOPE_GEOMETRY_NAMES
    )
    priors = "  ".join(
        f"{name.replace('morphology_', '').replace('_prior', '')}="
        f"{condition[names.index(name)].item():.3f}"
        for name in MORPHOLOGY_PRIOR_NAMES
    )
    ranges = env.haa_swing_ranges[robot_index].detach().cpu()
    range_text = "  ".join(
        f"{env.dof_names[index]}=[{ranges[leg, 0]:.3f}, {ranges[leg, 1]:.3f}]"
        for leg, index in enumerate(env.haa_indices.tolist())
    )
    print(f"\n[{label}]\n  {geometry}\n  priors: {priors}\n  HAA ranges: {range_text}")


def _update_follow_camera(env, camera_state, robot_index=0, mode="rear_side"):
    if getattr(env, "viewer", None) is None:
        return camera_state

    base_pos = env.root_states[robot_index, :3].detach().cpu().numpy()
    x, y, z, w = env.root_states[robot_index, 3:7].detach().cpu().numpy()
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if camera_state is None:
        camera_state = {"yaw": yaw, "base_z": base_pos[2], "position": None, "lookat": None}
    else:
        yaw_delta = np.arctan2(np.sin(yaw - camera_state["yaw"]), np.cos(yaw - camera_state["yaw"]))
        camera_state["yaw"] += 0.08 * yaw_delta
        camera_state["base_z"] += 0.03 * (base_pos[2] - camera_state["base_z"])

    stable_pos = base_pos.copy()
    stable_pos[2] = camera_state["base_z"]
    forward = np.array([np.cos(camera_state["yaw"]), np.sin(camera_state["yaw"]), 0.0])
    left = np.array([-forward[1], forward[0], 0.0])
    if mode == "rear_top":
        target_position = stable_pos - 1.25 * forward + np.array([0.0, 0.0, 1.8])
        target_lookat = stable_pos + 0.2 * forward + np.array([0.0, 0.0, 0.15])
    else:
        target_position = stable_pos - 1.6 * forward + 0.75 * left + np.array([0.0, 0.0, 0.65])
        target_lookat = stable_pos + 0.25 * forward + np.array([0.0, 0.0, 0.2])

    if camera_state["position"] is None:
        camera_state["position"] = target_position
        camera_state["lookat"] = target_lookat
    else:
        camera_state["position"] += 0.12 * (target_position - camera_state["position"])
        camera_state["lookat"] += 0.12 * (target_lookat - camera_state["lookat"])
    env.set_camera(camera_state["position"], camera_state["lookat"])
    return camera_state


def _validate_observation(obs, env):
    configured_dim = int(env.cfg.env.num_observations)
    if configured_dim != EXPECTED_OBS_DIM or obs.shape[-1] != EXPECTED_OBS_DIM:
        raise RuntimeError(
            f"envelop_2 observation mismatch: config={configured_dim}, "
            f"actual={obs.shape[-1]}, expected={EXPECTED_OBS_DIM}"
        )


def play(args):
    if args.task != TASK_NAME:
        print(f"play_envelop_2.py 使用任务 {TASK_NAME!r}（忽略传入的 {args.task!r}）")
        args.task = TASK_NAME

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1e9
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e9

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    env.commands[:, :3] = torch.tensor([1.2, 0.0, 0.0], device=env.device)
    active_condition = _apply_envelope(env, _maximum_envelope_condition(env))
    env.compute_observations()
    obs = env.get_observations()
    _validate_observation(obs, env)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        export_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.policy, export_path)
        print("Exported policy as jit script to:", export_path)

    command = {"x": 1.2, "y": 0.0, "yaw": 0.0}
    morphology_cycle = _new_morphology_cycle(env, diverse=True) if args.cycle_envelope else None
    camera_mode_index = 0
    camera_state = None
    if FOLLOW_CAMERA:
        camera_state = _update_follow_camera(env, camera_state, mode=CAMERA_MODES[camera_mode_index])

    _print_keyboard_table(FOLLOW_CAMERA)
    _print_envelope_state(env, "初始最大包络")
    print(f"\nObservation: {obs.shape[-1]} dims; 初始速度 vx={command['x']:.1f}")
    if morphology_cycle is not None:
        active_condition = _apply_envelope(env, morphology_cycle["start"])
        _print_envelope_state(env, "多样包络序列开始")

    keyboard = KeyboardInput()
    keyboard.__enter__()
    should_exit = False
    try:
        if keyboard.fd is None:
            print("stdin 不是 TTY，键盘控制已禁用。")

        total_steps = int(env.max_episode_length * 10)
        if args.max_steps is not None:
            if args.max_steps < 1:
                raise ValueError("--max_steps must be at least 1")
            total_steps = args.max_steps

        completed_steps = 0
        for _ in range(total_steps):
            step_start = time.time()
            for key in keyboard.read_keys():
                if key == "\x1b":
                    should_exit = True
                elif key == "w":
                    command["x"] = 1.2
                elif key == "s":
                    command["x"] = -1.2
                elif key == "a":
                    command["y"] = 0.8
                elif key == "d":
                    command["y"] = -0.8
                elif key == "q":
                    command["yaw"] = 0.8
                elif key == "e":
                    command["yaw"] = -0.8
                elif key == "x":
                    command = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                elif key == "m":
                    morphology_cycle = None
                    active_condition = _apply_envelope(env, _maximum_envelope_condition(env))
                    _print_envelope_state(env, "最大包络")
                elif key == "r":
                    morphology_cycle = None
                    active_condition = _apply_envelope(env, _random_envelope_condition(env))
                    _print_envelope_state(env, "随机包络")
                elif key == "f":
                    morphology_cycle = None
                    active_condition = _apply_envelope(env, _front_mammal_condition(env))
                    _print_envelope_state(env, "前腿 mammal 测试包络")
                elif key == "o":
                    morphology_cycle = _new_morphology_cycle(env)
                    active_condition = _apply_envelope(env, morphology_cycle["insect"])
                    _print_envelope_state(env, "O 循环开始：全昆虫型")
                elif key == "c" and FOLLOW_CAMERA:
                    camera_mode_index = (camera_mode_index + 1) % len(CAMERA_MODES)
                    camera_state["position"] = None
                    print(f"\n摄像头视角: {CAMERA_MODE_NAMES[CAMERA_MODES[camera_mode_index]]}")

            if should_exit:
                break

            if morphology_cycle is not None:
                cycle_step = morphology_cycle["step"]
                half_steps = morphology_cycle["half_steps"]
                if morphology_cycle["diverse"]:
                    active_condition = _apply_envelope(
                        env,
                        _transition_condition(
                            morphology_cycle["start"],
                            morphology_cycle["target"],
                            cycle_step,
                            half_steps,
                        ),
                        reset_transition=False,
                    )
                    if cycle_step >= half_steps:
                        _print_envelope_state(
                            env,
                            f"多样包络目标 {morphology_cycle['target_index']}",
                        )
                        morphology_cycle["start"] = morphology_cycle["target"]
                        morphology_cycle["target"] = _random_envelope_condition(env)
                        morphology_cycle["target_index"] += 1
                        morphology_cycle["step"] = 0
                    else:
                        morphology_cycle["step"] += 1
                else:
                    active_condition = _apply_envelope(
                        env,
                        _morphology_cycle_condition(
                            morphology_cycle["insect"],
                            morphology_cycle["mammal"],
                            cycle_step,
                            half_steps,
                        ),
                        reset_transition=False,
                    )
                    if cycle_step == half_steps:
                        _print_envelope_state(env, "O 循环中点：全哺乳型")
                    if cycle_step >= 2 * half_steps:
                        _print_envelope_state(env, "O 循环完成：全昆虫型")
                        morphology_cycle = None
                    else:
                        morphology_cycle["step"] += 1

            env.commands[:, 0] = command["x"]
            env.commands[:, 1] = command["y"]
            env.commands[:, 2] = command["yaw"]
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            completed_steps += 1
            if torch.any(dones):
                active_condition = _apply_envelope(env, active_condition)
                env.compute_observations()
                obs = env.get_observations()
            _validate_observation(obs, env)

            if FOLLOW_CAMERA:
                camera_state = _update_follow_camera(
                    env,
                    camera_state,
                    mode=CAMERA_MODES[camera_mode_index],
                )
            if REALTIME_MODE:
                sleep_time = env.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        print(f"Completed playback steps: {completed_steps}")
    finally:
        keyboard.__exit__(None, None, None)


if __name__ == "__main__":
    EXPORT_POLICY = False
    args = get_args(
        [
            {
                "name": "--max_steps",
                "type": int,
                "help": "Stop after this many simulation steps (default: interactive run).",
            },
            {
                "name": "--cycle_envelope",
                "action": "store_true",
                "default": False,
                "help": "Continuously interpolate through diverse random envelope targets.",
            },
        ]
    )
    FOLLOW_CAMERA = not args.headless
    REALTIME_MODE = not args.headless
    play(args)
