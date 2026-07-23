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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # Normalize terrain_proportions from config to ensure they sum to 1
        raw = np.array(cfg.terrain_proportions, dtype=float)
        s = raw.sum()
        if s == 0:
            # avoid division by zero: fallback to uniform distribution
            normalized = np.ones_like(raw) / len(raw)
        else:
            normalized = raw / s
        # store raw and normalized versions for debugging/inspection
        self.raw_proportions = raw.tolist()
        self.normalized_proportions = normalized.tolist()
        # cumulative probabilities used to select terrain types
        self.proportions = list(np.cumsum(normalized))

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            if hasattr(cfg, 'difficulty_scale'):
                self.curiculum(cfg.difficulty_scale)
            else:
                self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            if hasattr(cfg, 'difficulty_scale'):
                self.randomized_terrain(cfg.difficulty_scale)
            else:
                self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)
        print("Terrain proportions (raw):", getattr(self, 'raw_proportions', None),
              "normalized:", getattr(self, 'normalized_proportions', None),
              "cumulative:", self.proportions)

    def randomized_terrain(self, difficulty_scale=1.0):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9]) * difficulty_scale
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self, difficulty_scale=1.0):
        print("Generating curriculum terrains with difficulty scale: ", difficulty_scale)
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows * difficulty_scale
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.02 + difficulty * 0.2


        stepping_stones_size = self.cfg.stepping_stones_size + difficulty * 0.5
        stepping_stones_max_height = self.cfg.stepping_stones_max_height + difficulty * 0.3
        stepping_stones_platform_size = self.cfg.stepping_stones_platform_size
        stone_distance = self.cfg.stepping_stones_distance + difficulty * 0.5


        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height,
                                                     rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=stepping_stones_max_height, platform_size=stepping_stones_platform_size)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            pillar_field_terrain(terrain, difficulty, self.cfg)
        elif choice < self.proportions[8]:
            sin_curve_channel_terrain(terrain, difficulty, self.cfg)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2: center_x + x2, center_y-y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1: center_x + x1, center_y-y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def sin_curve_channel_terrain(terrain, difficulty, cfg):
    """Generate a sine-wave channel with walls on both sides.

    Channel runs along terrain width (world x-axis).  The centre varies
    along terrain length (world y-axis):

        centre = L/2 + amplitude * sin(2*pi*periods * world_pos / W_world)

    Parameters (from cfg):
        channel_width:   passable channel width (m).  Default 3.0.
        wall_height:     wall height (m).              Default 2.0.
        curve_amplitude: sine amplitude (m).           Default 1.0.
        curve_periods:   number of full sine periods.  Default 1.5.
    """
    channel_width = float(getattr(cfg, "channel_width", 3.0))
    wall_height = float(getattr(cfg, "wall_height", 2.0))
    curve_amplitude = float(getattr(cfg, "curve_amplitude", 1.0))
    curve_periods = float(getattr(cfg, "curve_periods", 1.5))

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    L = terrain.length
    W = terrain.width
    L_world = L * hs
    W_world = W * hs

    wall_px = int(wall_height / vs)
    half_px = int(channel_width / 2.0 / hs)

    for py in range(W):
        world_pos = py * hs
        centre = L_world / 2.0 + curve_amplitude * np.sin(
            2.0 * np.pi * curve_periods * world_pos / W_world)
        centre_px = int(centre / hs)
        lo = min(L, max(0, centre_px - half_px))
        hi = max(0, min(L, centre_px + half_px))
        terrain.height_field_raw[:lo, py] = wall_px
        terrain.height_field_raw[hi:, py] = wall_px

def pillar_field_terrain(terrain, difficulty, cfg):
    """
    生成随机分布的四棱柱障碍物地形（矩形截面）。

    预生成全部尺寸后逐个放置：每个障碍物在环带内随机采样位置，
    通过占用网格做 O(1) 碰撞检测（AABB + margin），兼顾效率和正确性。
    """
    # 数量范围（随难度插值）
    count_min = getattr(cfg, "pillar_count_min", 0)
    count_max = getattr(cfg, "pillar_count_max", 25)
    # 矩形边长范围（米）
    size_x_min = getattr(cfg, "pillar_size_x_min", 0.15)
    size_x_max = getattr(cfg, "pillar_size_x_max", 0.30)
    size_y_min = getattr(cfg, "pillar_size_y_min", 0.15)
    size_y_max = getattr(cfg, "pillar_size_y_max", 0.30)
    # 高度范围（米）
    height_min = getattr(cfg, "pillar_height_min", 0.20)
    height_max = getattr(cfg, "pillar_height_max", 0.50)
    # 间距与放置
    min_separation = getattr(cfg, "pillar_min_separation", 1.2)
    center_clear_radius = getattr(cfg, "pillar_center_clear_radius", 1.2)
    spawn_radius = getattr(cfg, "pillar_spawn_radius", 4.0)
    allow_height_variation = getattr(cfg, "pillar_allow_height_variation", True)

    count = int(count_min + difficulty * (count_max - count_min))

    min_sep_px = int(min_separation / terrain.horizontal_scale)
    clear_radius_px = int(center_clear_radius / terrain.horizontal_scale)
    spawn_radius_px = int(spawn_radius / terrain.horizontal_scale)
    margin = min_sep_px // 2

    L = terrain.length
    W = terrain.width
    center_y = L // 2
    center_x = W // 2

    # ---- 预生成全部尺寸（长/短边 split + 随机方向） ----
    half_x_px = []
    half_y_px = []
    occ_hx_px = []
    occ_hy_px = []
    height_px = []

    for _ in range(count):
        split = np.random.uniform(0.2, 0.8)
        x_range = size_x_max - size_x_min
        y_range = size_y_max - size_y_min

        if np.random.rand() > 0.5:
            long_min = size_x_min + split * x_range
            sx = long_min + np.random.uniform(0.0, 1.0) * (size_x_max - long_min)
            short_max = size_y_min + split * y_range
            sy = size_y_min + np.random.uniform(0.0, 1.0) * (short_max - size_y_min)
        else:
            long_min = size_y_min + split * y_range
            sy = long_min + np.random.uniform(0.0, 1.0) * (size_y_max - long_min)
            short_max = size_x_min + split * x_range
            sx = size_x_min + np.random.uniform(0.0, 1.0) * (short_max - size_x_min)

        hx = int(sx / terrain.horizontal_scale) // 2
        hy = int(sy / terrain.horizontal_scale) // 2
        h_val = height_min + np.random.uniform(0.0, 1.0) * (height_max - height_min)

        half_x_px.append(hx)
        half_y_px.append(hy)
        occ_hx_px.append(hx + margin)
        occ_hy_px.append(hy + margin)
        height_px.append(int(h_val / terrain.vertical_scale))

    # ---- 占用网格 ----
    occupied = np.zeros((L, W), dtype=bool)

    max_attempts_per = count * 100
    positions = []

    for i in range(count):
        hx = half_x_px[i]
        hy = half_y_px[i]
        occ_hx = occ_hx_px[i]
        occ_hy = occ_hy_px[i]

        # 障碍物太大跳过
        if hy * 2 >= L or hx * 2 >= W:
            continue

        for _ in range(max_attempts_per):
            r = np.random.uniform(clear_radius_px, spawn_radius_px)
            theta = np.random.uniform(0, 2 * np.pi)
            cx = int(center_x + r * np.cos(theta))
            cy = int(center_y + r * np.sin(theta))

            # 边界检查（仅本体）
            if cx - hx < 0 or cx + hx >= W or cy - hy < 0 or cy + hy >= L:
                continue

            # 中心点距离检查
            if np.hypot(cx - center_x, cy - center_y) < clear_radius_px:
                continue

            # O(1) 碰撞检测：占用网格切片
            oy1 = cy - occ_hy
            oy2 = cy + occ_hy
            ox1 = cx - occ_hx
            ox2 = cx + occ_hx
            if oy1 < 0 or oy2 > L or ox1 < 0 or ox2 > W:
                continue
            if occupied[oy1:oy2, ox1:ox2].any():
                continue

            # 放置成功
            occupied[oy1:oy2, ox1:ox2] = True
            positions.append((cx, cy, height_px[i], hx, hy))
            break

    # ---- 绘制 ----
    for cx, cy, h_px, hx, hy in positions:
        if allow_height_variation:
            h = np.random.randint(int(h_px * 0.6), h_px + 1)
        else:
            h = h_px

        x1 = max(0, cx - hx)
        x2 = min(W, cx + hx)
        y1 = max(0, cy - hy)
        y2 = min(L, cy + hy)
        terrain.height_field_raw[x1:x2, y1:y2] = h

    return terrain