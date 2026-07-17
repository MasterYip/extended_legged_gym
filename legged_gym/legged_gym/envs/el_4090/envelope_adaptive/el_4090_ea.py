import math
import torch
import numpy as np
import warp as wp

from isaacgym.torch_utils import (
    quat_apply, quat_mul, quat_from_euler_xyz, quat_from_angle_axis,
    torch_rand_float,
)
from isaacgym import gymapi, gymtorch
import isaacgym.gymutil as gymutil
import matplotlib.pyplot as plt

from legged_gym.envs.el_4090.spider_nomal.el_4090 import EL_4090
from legged_gym.utils.math_utils import quat_apply_yaw
from legged_gym.utils.LidarSensor.LidarSensor.lidar_sensor import LidarSensor
from legged_gym.utils.LidarSensor.LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

from legged_gym.envs.el_4090.envelope_adaptive.envelope_computer import (
    compute_envelope_params, envelope_params_to_condition, ENVELOPE_PARAM_NAMES,
    _build_hex_edges, _offset_hexagon, _make_min_hex,
)
from legged_gym.envs.el_4090.envelope_adaptive.avoidance_planner import compute_safe_velocity


class EL_4090_EA(EL_4090):
    """EL_4090 with Envelope Adaptive obstacle avoidance.

    LiDAR point cloud processed externally -- no dual-GRU network.
    Policy receives 74-dim observation (66 proprio + 8 envelope condition).
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 task_name="el_4090_ea"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         task_name=task_name)
        self._lidar_decimation = max(1, round(1.0 / (self.dt * self.cfg.raycaster.update_frequency_hz)))
        self._lidar_timer = self._lidar_decimation - 1  # 固定时钟，首帧立即触发

    # ==================================================================
    # Buffer initialisation
    # ==================================================================

    def _init_buffers(self):
        super()._init_buffers()
        self._init_lidar_buffers()
        self._init_lidar()
        self._init_condition_buffers()

    def _init_lidar_buffers(self):
        cfg = self.cfg.raycaster
        n_pts = int(cfg.spherical_num_azimuth * cfg.spherical_num_elevation)
        d_max = float(cfg.max_distance)

        self.lidar_points_base = torch.zeros(
            self.num_envs, n_pts, 3, device=self.device,
            dtype=torch.float, requires_grad=False,
        )
        self.raycast_distances = torch.full(
            (self.num_envs, n_pts), d_max, device=self.device,
            dtype=torch.float, requires_grad=False,
        )
        # Envelope params buffer -- start at max envelope, shrink on obstacle detection
        # 5 个对称参数,键名 = condition 名;边界 = cfg.commands.ranges(单一事实来源)
        ranges = self.cfg.commands.ranges
        self.env_params = {
            "front_width": torch.full((self.num_envs,), ranges.front_width[1], device=self.device),
            "middle_width": torch.full((self.num_envs,), ranges.middle_width[1], device=self.device),
            "back_width": torch.full((self.num_envs,), ranges.back_width[1], device=self.device),
            "forward_limit": torch.full((self.num_envs,), ranges.forward_limit[1], device=self.device),
            "backward_limit": torch.full((self.num_envs,), ranges.backward_limit[0], device=self.device),
        }
        # per-param grow cooldown counter (E, 5)
        self._grow_cooldown = torch.zeros(
            self.num_envs, len(ENVELOPE_PARAM_NAMES),
            dtype=torch.int, device=self.device,
        )
        # pre-built minimum hexagon for detection band lower bound
        self._min_hex = _make_min_hex(self.cfg, device=self.device)  # (1, 6, 2)

    def _init_condition_buffers(self):
        """初始化包络 condition 相关 buffer（对齐底层 spider_envelop el_4090.py:100-200）"""
        # condition 在 commands buffer 中的切片位置
        self.condition_start_idx = 4
        self.condition_names = list(self.cfg.commands.condition_names)
        self.condition_dim = int(self.cfg.commands.condition_dim)
        self.condition_end_idx = self.condition_start_idx + self.condition_dim

        # condition 上下界 tensor（由 command_ranges 构建）
        self.condition_low = torch.tensor(
            [self.command_ranges[name][0] for name in self.condition_names],
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.condition_high = torch.tensor(
            [self.command_ranges[name][1] for name in self.condition_names],
            dtype=torch.float, device=self.device, requires_grad=False,
        )

        # mammal 默认关节角
        mammal_angles = self.cfg.init_state.mammal_default_joint_angles
        self.mammal_default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False,
        )
        for i, name in enumerate(self.dof_names):
            self.mammal_default_dof_pos[i] = mammal_angles[name]
        self.mammal_default_dof_pos = self.mammal_default_dof_pos.unsqueeze(0)  # (1, 18)

        # morphology prior → 腿组 DOF 索引映射（对齐底层 el_4090.py:163-192）
        self.morphology_prior_leg_groups = {
            "morphology_front_prior": ["LF", "RF"],
            "morphology_middle_prior": ["LM", "RM"],
            "morphology_back_prior": ["LB", "RB"],
        }
        self.morphology_prior_dof_indices = {}
        for prior_name, leg_names in self.morphology_prior_leg_groups.items():
            indices = [
                i for i, dof_name in enumerate(self.dof_names)
                if any(dof_name.startswith(f"{leg_name}_") for leg_name in leg_names)
            ]
            assert len(indices) == 3 * len(leg_names), \
                f"Expected {3 * len(leg_names)} DOFs for {prior_name}, got {len(indices)}"
            self.morphology_prior_dof_indices[prior_name] = torch.tensor(
                indices, dtype=torch.long, device=self.device, requires_grad=False,
            )

        # embedded_state DOF target 及 P_LOWPASS 滤波器状态
        self.embedded_state_default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.filtered_embedded_state_default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        # 观测缩放因子（对齐底层 el_4090.py:100-102）
        self.command_lin_vel_scale = torch.tensor(
            self.obs_scales.lin_vel, device=self.device, requires_grad=False,
        )
        self.command_ang_vel_scale = torch.tensor(
            self.obs_scales.ang_vel, device=self.device, requires_grad=False,
        )
        self.command_embedded_state_scale = torch.tensor(
            self.obs_scales.embedded_state, device=self.device, requires_grad=False,
        )

        # 最大包络对应的 condition 常量(含 3 先验 ≈0.588/0.235/0.588),
        # 供 reset_idx 写入与 reset 高度插值两处复用
        max_params = {k: v[:1].clone() for k, v in self.env_params.items()}
        self._max_envelope_condition = envelope_params_to_condition(max_params, self.cfg)  # (1, 8)

        # P5: reset 初始高度随形态先验插值(对齐底层 spider_envelop;
        # EA 中 reset 后包络恒为 max → 常量 ≈0.582)
        spider_h = float(getattr(self.cfg.rewards, "base_height_spider_target", 0.53))
        mammal_h = float(getattr(self.cfg.rewards, "base_height_mammal_target", spider_h))
        mean_prior = self._max_envelope_condition[0, 5:8].mean().item()
        self._reset_base_height = spider_h + mean_prior * (mammal_h - spider_h)

    def _get_noise_scale_vec(self, cfg):
        """74-dim noise vector matching spider_envelop observation layout."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:20] = 0.0   # commands (3 vel + 8 condition)
        noise_vec[20:38] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[38:56] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[56:74] = 0.0  # previous actions
        return noise_vec

    # ==================================================================
    # Observations, torques, and helpers
    # ==================================================================

    def compute_observations(self):
        """74-dim observation matching spider_envelop layout (el_4090.py:1039-1052)."""
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,                            # 0:3
            self.base_ang_vel * self.obs_scales.ang_vel,                            # 3:6
            self.projected_gravity,                                                  # 6:9
            self.commands[:, 0:1] * self.command_lin_vel_scale,                      # 9
            self.commands[:, 1:2] * self.command_lin_vel_scale,                      # 10
            self.commands[:, 2:3] * self.command_ang_vel_scale,                      # 11
            self.commands[:, self.condition_start_idx:self.condition_end_idx]
                * self.command_embedded_state_scale,                                 # 12:20 (8 dims)
            (self.dof_pos - self.embedded_state_default_dof_pos)
                * self.obs_scales.dof_pos,                                           # 20:38 (18 dims)
            self.dof_vel * self.obs_scales.dof_vel,                                  # 38:56 (18 dims)
            self.actions,                                                            # 56:74 (18 dims)
        ), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        """PD controller with P_LOWPASS mode for smooth morphology transition.

        P_LOWPASS 之外的控制类型(P/V/T/DELAY)委托给父类实现。
        """
        control_type = self.cfg.control.control_type
        if control_type != "P_LOWPASS":
            return super()._compute_torques(actions)

        actions_scaled = actions * self.cfg.control.action_scale
        # guard against tau=0 division by zero
        tau = max(float(getattr(self.cfg.control, "default_dof_pos_filter_tau", 0.3)), 1e-6)
        alpha = 1.0 - np.exp(-self.sim_params.dt / tau)
        self.filtered_embedded_state_default_dof_pos += alpha * (
            self.embedded_state_default_dof_pos - self.filtered_embedded_state_default_dof_pos
        )
        done_threshold = float(getattr(self.cfg.control,
                                       "default_dof_pos_filter_done_threshold", 0.02))
        filter_error = torch.max(
            torch.abs(self.filtered_embedded_state_default_dof_pos
                      - self.embedded_state_default_dof_pos),
            dim=1,
        ).values
        use_final_default = (filter_error < done_threshold).unsqueeze(1)
        smoothed_default_dof_pos = torch.where(
            use_final_default,
            self.embedded_state_default_dof_pos,
            self.filtered_embedded_state_default_dof_pos,
        )
        torques = (self.p_gains * (actions_scaled + smoothed_default_dof_pos - self.dof_pos)
                   - self.d_gains * self.dof_vel)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_structure_condition(self):
        """读取 condition 并 clip 到上下界（对齐底层 el_4090.py:1066-1068）"""
        condition = self.commands[:, self.condition_start_idx:self.condition_end_idx]
        return torch.minimum(torch.maximum(condition, self.condition_low), self.condition_high)

    def _get_morphology_prior(self, prior_name):
        """获取指定 morphology prior 的值（对齐底层 el_4090.py:1187-1191）"""
        if prior_name not in self.condition_names:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        prior_idx = self.condition_names.index(prior_name)
        return self._get_structure_condition()[:, prior_idx]

    def _get_condition_target_dof_pos(self):
        """按 morphology prior 在 spider/mammal 默认关节角间插值（对齐底层 el_4090.py:1218-1225）"""
        target = self.default_dof_pos.repeat(self.num_envs, 1)
        for prior_name, dof_indices in self.morphology_prior_dof_indices.items():
            prior = self._get_morphology_prior(prior_name).unsqueeze(1)
            spider_pos = self.default_dof_pos[:, dof_indices]
            mammal_pos = self.mammal_default_dof_pos[:, dof_indices]
            target[:, dof_indices] = spider_pos + prior * (mammal_pos - spider_pos)
        return target

    # ==================================================================
    # LiDAR sensor (simplified -- no sector safety, no proximal/distal)
    # ==================================================================

    def _init_lidar(self):
        if not getattr(self.cfg.raycaster, "enable_raycast", False):
            self.lidar_sensor = None
            return

        wp.init()

        vertices, triangles_i32 = self._build_warp_mesh()
        self._wp_mesh = wp.Mesh(
            points=wp.from_torch(vertices, dtype=wp.vec3),
            indices=wp.from_numpy(triangles_i32.flatten(), dtype=wp.int32,
                                  device=self.device),
        )
        self.mesh_ids = wp.array([self._wp_mesh.id], dtype=wp.uint64)

        self.sensor_pos_tensor = torch.zeros_like(self.base_pos)
        self.sensor_quat_tensor = torch.zeros_like(self.base_quat)

        ray_cfg = self.cfg.raycaster
        update_hz = float(getattr(ray_cfg, "update_frequency_hz",
                                  max(1.0, 1.0 / float(self.dt))))
        lidar_cfg = LidarConfig(
            sensor_type=LidarType.AIRY,
            dt=float(self.dt),
            update_frequency=update_hz,
            max_range=float(ray_cfg.max_distance),
            min_range=0.1,                    # Airy 盲区 < 0.1m
            num_sensors=1,
            horizontal_line_num=int(ray_cfg.spherical_num_azimuth),
            vertical_line_num=int(ray_cfg.spherical_num_elevation),
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=float(getattr(ray_cfg, "vertical_fov_deg_min", 0.0)),
            vertical_fov_deg_max=float(getattr(ray_cfg, "vertical_fov_deg_max", 90.0)),
            return_pointcloud=True,
            pointcloud_in_world_frame=False,
            randomize_placement=False,
            enable_sensor_noise=False,
        )

        lidar_env = {
            "device": self.device,
            "num_envs": self.num_envs,
            "num_sensors": 1,
            "sensor_pos_tensor": self.sensor_pos_tensor,
            "sensor_quat_tensor": self.sensor_quat_tensor,
            "mesh_ids": self.mesh_ids,
        }
        self.lidar_sensor = LidarSensor(lidar_env, None, lidar_cfg,
                                        num_sensors=1, device=self.device)

        # Sensor offset in body frame
        offset_pos = list(ray_cfg.offset_pos)
        self._sensor_translation = torch.tensor(
            offset_pos, dtype=torch.float32, device=self.device,
        ).view(1, 3).repeat(self.num_envs, 1)

        rpy = getattr(ray_cfg, "sensor_offset_rpy", [0.0, 0.0, 0.0])
        offset_q = quat_from_euler_xyz(
            torch.tensor(float(rpy[0]), device=self.device),
            torch.tensor(float(rpy[1]), device=self.device),
            torch.tensor(float(rpy[2]), device=self.device),
        )
        self._sensor_offset_quat = offset_q.view(1, 4).repeat(self.num_envs, 1)

    def _build_warp_mesh(self):
        """Build (vertices, triangles) for Warp raycasting."""
        terrain = self.cfg.terrain

        if (hasattr(self, "terrain")
                and hasattr(self.terrain, "vertices")
                and hasattr(self.terrain, "triangles")):
            vertices = torch.as_tensor(
                self.terrain.vertices, device=self.device, dtype=torch.float32,
            ).clone()
            if hasattr(terrain, "border_size"):
                vertices[:, 0] -= terrain.border_size
                vertices[:, 1] -= terrain.border_size
            triangles_i32 = np.asarray(self.terrain.triangles, dtype=np.int32)
        elif terrain.mesh_type == "plane":
            plane_size = float(getattr(terrain, "lidar_plane_size", 100.0))
            vertices = torch.tensor(
                [[-plane_size, -plane_size, 0.0],
                 [plane_size, -plane_size, 0.0],
                 [plane_size, plane_size, 0.0],
                 [-plane_size, plane_size, 0.0]],
                device=self.device, dtype=torch.float32,
            )
            triangles_i32 = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        else:
            raise ValueError(
                "EL_4090_EA requires trimesh terrain or plane mesh_type "
                "for LiDAR rendering."
            )
        return vertices, triangles_i32

    # ==================================================================
    # LiDAR update
    # ==================================================================

    def _update_lidar(self):
        """Update LiDAR point cloud for current step."""
        if self.lidar_sensor is None:
            return

        self._lidar_timer += 1
        if self._lidar_timer % self._lidar_decimation != 0:
            return  # 复用上一帧 lidar_points_base
        self._lidar_timer = 0

        self.sensor_quat_tensor.copy_(
            quat_mul(self.base_quat, self._sensor_offset_quat))
        self.sensor_pos_tensor.copy_(
            self.base_pos + quat_apply(self.base_quat, self._sensor_translation))

        lidar_points, lidar_dist = self.lidar_sensor.update()
        points_sensor = lidar_points.view(self.num_envs, -1, 3)
        n_points = points_sensor.shape[1]
        dist = lidar_dist.view(self.num_envs, -1)

        # Transform: sensor frame -> base frame
        quat_1x4 = self._sensor_offset_quat[0:1]
        n_total = int(points_sensor.numel() // 3)
        points_base = quat_apply(
            quat_1x4.expand(n_total, 4), points_sensor.reshape(-1, 3),
        ).reshape(self.num_envs, n_points, 3) + self._sensor_translation.unsqueeze(1)

        self.lidar_points_base.copy_(points_base)
        self.raycast_distances.copy_(dist)

    # ==================================================================
    # Envelope update + draw
    # ==================================================================

    def _update_envelope(self):
        """Compute envelope params, write condition to commands, and draw the 3D prism."""
        if self.lidar_sensor is None:
            return
        params, self._grow_cooldown = compute_envelope_params(
            self.lidar_points_base, self.base_pos, self.base_quat,
            self.cfg, self.env_params, self._grow_cooldown, self._min_hex,
        )
        for k, v in params.items():
            self.env_params[k].copy_(v)

        # 包络参数 → condition，连续流式写入 commands[:, 4:12]
        condition = envelope_params_to_condition(self.env_params, self.cfg)
        self.commands[:, self.condition_start_idx:self.condition_end_idx] = condition

        self._draw_envelope()

    def _draw_hex_prism(self, verts_top, verts_bot, env_id, color, rad):
        for hex_verts in [verts_top, verts_bot]:
            for i in range(6):
                j = (i + 1) % 6
                self.vis.draw_boldline(env_id, [
                    hex_verts[i].cpu().numpy().tolist(),
                    hex_verts[j].cpu().numpy().tolist(),
                ], rad=rad, color=color)
        for i in range(6):
            self.vis.draw_boldline(env_id, [
                verts_top[i].cpu().numpy().tolist(),
                verts_bot[i].cpu().numpy().tolist(),
            ], rad=rad, color=color)

    def _draw_lidar_points(self, eid, bp, bq, inner_verts, outer_verts, zt, zb):
        """Draw LiDAR point cloud: blue=valid hit, red=in 3D sensitive zone.

        inner_verts/outer_verts: (6, 2) — 由 _draw_envelope 构建并复用(与检测同一几何)。
        """
        pts = self.lidar_points_base[eid]                               # (N, 3) body-frame
        dists = self.raycast_distances[eid]                             # (N,)
        d_max = float(self.cfg.raycaster.max_distance)
        valid = dists < (d_max - 0.001)
        if not valid.any():
            return
        pts = pts[valid]
        pts_world = (bp.unsqueeze(0) + quat_apply(
            bq.unsqueeze(0).expand(pts.shape[0], 4), pts)).cpu().numpy()

        in_z = (pts[:, 2] >= zb) & (pts[:, 2] <= zt)

        def _inside(pts_xy, verts):
            V = 6
            nxt = (torch.arange(V) + 1) % V
            edges = verts[nxt, :] - verts
            rel = pts_xy.unsqueeze(1) - verts.unsqueeze(0)
            cross = edges[:, 0] * rel[..., 1] - edges[:, 1] * rel[..., 0]
            return (cross >= 0).all(dim=-1)

        pts_xy = pts[:, :2]
        in_zone = _inside(pts_xy, outer_verts) & ~_inside(pts_xy, inner_verts) & in_z
        in_zone_np = in_zone.cpu().numpy()

        red_pts = pts_world[in_zone_np]
        blue_pts = pts_world[~in_zone_np]

        red_geom = gymutil.WireframeSphereGeometry(0.04, 4, 4, None, color=(1, 0, 0))
        for pt in red_pts:
            pose = gymapi.Transform(gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2])))
            gymutil.draw_lines(red_geom, self.gym, self.viewer, self.envs[eid], pose)

        blue_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0.2, 0.4, 1))
        for pt in blue_pts:
            pose = gymapi.Transform(gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2])))
            gymutil.draw_lines(blue_geom, self.gym, self.viewer, self.envs[eid], pose)

    def _draw_envelope(self):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        zt = self.cfg.envelope.z_top
        zb = self.cfg.envelope.z_bottom
        m = self.cfg.envelope.margin_distance
        eid = 0
        bp = self.base_pos[eid]; bq = self.base_quat[eid]

        # inner/outer 2D 顶点只构建一次,绘制与红蓝分类复用(与检测同一几何)
        p = {k: self.env_params[k][eid:eid + 1] for k in self.env_params}
        inner_2d = _build_hex_edges(
            p["forward_limit"], p["backward_limit"],
            p["front_width"], p["middle_width"], p["back_width"],
        )                                                   # (1, 6, 2)
        outer_2d = _offset_hexagon(inner_2d, m)             # (1, 6, 2) 径向,P2 修复
        inner_v = inner_2d[0]
        outer_v = outer_2d[0]

        def _prism(verts_2d, z):
            zc = torch.full((6, 1), float(z), device=verts_2d.device)
            return torch.cat([verts_2d, zc], dim=-1)        # (6, 3)

        self._draw_hex_prism(bp + quat_apply(bq, _prism(inner_v, zt)),
                             bp + quat_apply(bq, _prism(inner_v, zb)),
                             eid, (0.0, 1.0, 0.5), 0.02)
        self._draw_hex_prism(bp + quat_apply(bq, _prism(outer_v, zt)),
                             bp + quat_apply(bq, _prism(outer_v, zb)),
                             eid, (0.3, 0.7, 0.5), 0.01)
        # point cloud(复用同一 inner/outer 顶点)
        self._draw_lidar_points(eid, bp, bq, inner_v, outer_v, zt, zb)

    # ==================================================================
    # Avoidance velocity (passthrough stub)
    # ==================================================================

    def _compute_avoidance_vel(self):
        if self.lidar_sensor is None:
            self._desired_cmd = self.commands[:, :3].clone()
            self._safe_cmd = self.commands[:, :3].clone()
            return
        self._desired_cmd = self.commands[:, :3].clone()
        cfg = getattr(self.cfg, "avoidance", self.cfg.envelope)
        prev_theta = getattr(self, '_prev_theta_out', None)
        safe_vx, safe_vy, self._avoid_debug = compute_safe_velocity(
            self.lidar_points_base, self.raycast_distances,
            self.base_pos, self.base_quat,
            self.commands, cfg, prev_theta=prev_theta,
        )
        if safe_vx.abs().sum() > 1e-6:
            self._prev_theta_out = float(np.arctan2(
                safe_vy[0].item(), safe_vx[0].item()))
        self.commands[:, 0] = safe_vx
        self.commands[:, 1] = safe_vy
        self._safe_cmd = self.commands[:, :3].clone()

    def _draw_velocity_arrows(self):
        """Draw desired (yellow) and safe (red) velocity arrows (world-frame)."""
        if self.viewer is None:
            return
        for i in range(self.num_envs):
            bp = self.root_states[i, :3].cpu().numpy()
            # desired
            d_world = quat_apply_yaw(self.base_quat[i:i+1],
                                     self._desired_cmd[i:i+1]).cpu().numpy()[0]
            self.vis.draw_arrow(i, bp, bp + d_world * 0.5, color=(1, 1, 0))
            # safe
            s_world = quat_apply_yaw(self.base_quat[i:i+1],
                                     self._safe_cmd[i:i+1]).cpu().numpy()[0]
            self.vis.draw_arrow(i, bp, bp + s_world * 0.5, color=(1, 0, 0))

    def _draw_avoidance_debug(self):
        """Matplotlib polar plot: angular-distance curve, spline, peaks."""
        if not self._avoid_debug:
            return
        if not hasattr(self, '_debug_fig'):
            plt.ion()
            self._debug_fig, self._debug_ax = plt.subplots(
                subplot_kw={'projection': 'polar'}, figsize=(5, 5))
            self._debug_step = 0
        self._debug_step += 1
        if self._debug_step % 10 != 0:
            return

        d = self._avoid_debug
        ax = self._debug_ax
        ax.clear()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(-180); ax.set_thetamax(180)

        theta = d["theta"]; d_raw = d["d_raw"]; d_smooth = d["d_smooth"]
        ax.scatter(theta, d_raw, s=10, c='blue', label='raw')
        ax.plot(theta, d_smooth, 'c-', linewidth=1, label='spline')

        for p_theta, p_d in d["peaks"]:
            ax.plot(p_theta, p_d, 'ro', markersize=6)
        if d["best_theta"] is not None:
            # find d at best theta from smooth
            bth = d["best_theta"]
            bd = np.interp(bth, theta, d_smooth)
            ax.plot(bth, bd, 'g*', markersize=12, label='selected')

        # cmd direction
        t_cmd = d["theta_cmd"]
        ax.plot([t_cmd, t_cmd], [0, 10], 'm--', linewidth=1, label='cmd')

        ax.set_ylim(0, d_raw.max() + 0.5)
        ax.legend(loc='lower right', fontsize=6)
        self._debug_fig.canvas.draw_idle()
        plt.pause(0.001)

    # ==================================================================
    # Step hooks
    # ==================================================================

    def post_physics_step(self):
        super().post_physics_step()
        self._update_lidar()
        self._update_envelope()                 # ← 内部已将 condition 写入 commands[:, 4:12]
        # 每步更新形态插值目标（condition 变化后立即刷新）
        self.embedded_state_default_dof_pos = self._get_condition_target_dof_pos()
        self._compute_avoidance_vel()
        self.compute_observations()             # ← 使用最新的 commands + embedded target
        self._draw_velocity_arrows()
        self._draw_avoidance_debug()

    # ==================================================================
    # Reset
    # ==================================================================

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)

        # LiDAR buffer reset
        d_max = float(self.cfg.raycaster.max_distance)
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = d_max

        # 重置包络参数到最大包络端(防止前一 episode 收缩后的参数泄漏到新 episode)
        ranges = self.cfg.commands.ranges
        self.env_params["front_width"][env_ids] = ranges.front_width[1]
        self.env_params["middle_width"][env_ids] = ranges.middle_width[1]
        self.env_params["back_width"][env_ids] = ranges.back_width[1]
        self.env_params["forward_limit"][env_ids] = ranges.forward_limit[1]
        self.env_params["backward_limit"][env_ids] = ranges.backward_limit[0]
        # condition 立即写入最大包络值(正常路径下同一 post_physics_step 尾部的
        # _update_envelope 会以相同值覆盖;价值在 lidar 禁用调试路径与防御性鲁棒)
        self.commands[env_ids, self.condition_start_idx:self.condition_end_idx] = \
            self._max_envelope_condition
        # P_LOWPASS 滤波器从 default_dof_pos 重新开始平滑
        self.filtered_embedded_state_default_dof_pos[env_ids] = self.default_dof_pos
        # cooldown 重置为 0 (reset 后包络在 max, 无需立即 grow)
        self._grow_cooldown[env_ids] = 0

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # P5: 初始 base 高度随形态先验插值(对齐底层 el_4090.py:1028-1036;
        # EA 在唯一一次 gym tensor 写入前替换 z,省去底层的二次 API 调用)
        if getattr(self.cfg.rewards, "reset_base_height_with_morphology", False):
            self.root_states[env_ids, 2] = (
                self.env_origins[env_ids, 2] + self._reset_base_height
            )

        spawn_range = float(getattr(self.cfg.init_state, "spawn_offset_range", 0.5))
        self.root_states[env_ids, :2] += torch_rand_float(
            -spawn_range, spawn_range, (len(env_ids), 2), device=self.device)

        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device)

        if self.cfg.init_state.randomize_rot:
            r0, r1 = self.cfg.init_state.rot_randomization_range
            rand_yaw = torch_rand_float(
                r0, r1, (len(env_ids), 1), device=self.device).squeeze(1)
            axis = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
            self.root_states[env_ids, 3:7] = quat_from_angle_axis(rand_yaw, axis)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # ==================================================================
    # Debug viz suppressed (envelope drawn separately)
    # ==================================================================

    def draw_foot_hip_positions(self):
        """Suppressed: clear_lines in _draw_envelope replaces debug viz."""
        pass
    