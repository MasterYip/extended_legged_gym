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

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt, output_root: Path = None):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.output_root = Path(output_root) if output_root is not None else None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew_' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _save_vector_plots(self, fig, log, time):
        if self.output_root is not None:
            output_dir = self.output_root / "picture"
        else:
            output_dir = Path(__file__).resolve().parent / "picutres"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _save_single_plot(name, plot_func):
            single_fig, single_ax = plt.subplots(figsize=(6, 4))
            plot_func(single_ax)
            handles, labels = single_ax.get_legend_handles_labels()
            if labels:
                single_ax.legend()
            single_fig.tight_layout()
            single_fig.savefig(output_dir / f"{timestamp}_{name}.svg", format='svg')
            plt.close(single_fig)

        _save_single_plot(
            "base_vel_x",
            lambda a: (
                a.plot(time, log["base_vel_x"], label='measured') if log["base_vel_x"] else None,
                a.plot(time, log["command_x"], label='commanded') if log["command_x"] else None,
                a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
            )
        )
        _save_single_plot(
            "base_vel_y",
            lambda a: (
                a.plot(time, log["base_vel_y"], label='measured') if log["base_vel_y"] else None,
                a.plot(time, log["command_y"], label='commanded') if log["command_y"] else None,
                a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
            )
        )
        _save_single_plot(
            "base_vel_yaw",
            lambda a: (
                a.plot(time, log["base_vel_yaw"], label='measured') if log["base_vel_yaw"] else None,
                a.plot(time, log["command_yaw"], label='commanded') if log["command_yaw"] else None,
                a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
            )
        )
        _save_single_plot(
            "dof_pos",
            lambda a: (
                a.plot(time, log["dof_pos"], label='measured') if log["dof_pos"] else None,
                a.plot(time, log["dof_pos_target"], label='target') if log["dof_pos_target"] else None,
                a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
            )
        )
        _save_single_plot(
            "dof_vel",
            lambda a: (
                a.plot(time, log["dof_vel"], label='measured') if log["dof_vel"] else None,
                a.plot(time, log["dof_vel_target"], label='target') if log["dof_vel_target"] else None,
                a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
            )
        )
        _save_single_plot(
            "joint_vel_compare",
            lambda a: (
                a.plot(time, log["dof_vel"], label='dof_vel') if log["dof_vel"] != [] else None,
                a.plot(time, log["dof_vel_1"], label='dof_vel_1') if log["dof_vel_1"] != [] else None,
                a.plot(time, log["dof_vel_2"], label='dof_vel_2') if log["dof_vel_2"] != [] else None,
                a.set(xlabel='time [s]', ylabel='Joint vel [rad/s]', title='Joint Velocity')
            )
        )

        def _plot_contact_forces(a):
            if log["contact_forces_z"]:
                forces = np.array(log["contact_forces_z"])
                for i in range(forces.shape[1]):
                    a.plot(time, forces[:, i], label=f'force {i}')
            a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')

        _save_single_plot("contact_forces_z", _plot_contact_forces)

        _save_single_plot(
            "torque_velocity",
            lambda a: (
                a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured') if log["dof_vel"] != [] and log["dof_torque"] != [] else None,
                a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
            )
        )
        _save_single_plot(
            "torque",
            lambda a: (
                a.plot(time, log["dof_torque_1"], label='dof_torque_1') if log["dof_torque_1"] != [] else None,
                a.plot(time, log["dof_torque_2"], label='dof_torque_2') if log["dof_torque_2"] != [] else None,
                a.plot(time, log["dof_torque"], label='dof_torque') if log["dof_torque"] != [] else None,
                a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
            )
        )

        print(f"Saved vector plots to: {output_dir}")

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        if len(self.state_log) == 0:
            return
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        # if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        if log["dof_vel"]!=[]: a.plot(time, log["dof_vel"], label='dof_vel')
        if log["dof_vel_1"]!=[]: a.plot(time, log["dof_vel_1"], label='dof_vel_1')
        if log["dof_vel_2"]!=[]: a.plot(time, log["dof_vel_2"], label='dof_vel_2')
        a.set(xlabel='time [s]', ylabel='Joint vel [rad/s]', title='Joint Velocity')

        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque_1"]!=[]: a.plot(time, log["dof_torque_1"], label='dof_torque_1')
        if log["dof_torque_2"]!=[]: a.plot(time, log["dof_torque_2"], label='dof_torque_2')
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='dof_torque')

        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        self._save_vector_plots(fig, log, time)
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()