import math
import torch
import numpy as np
import warp as wp

from isaacgym.torch_utils import (
    quat_apply, quat_mul, quat_from_euler_xyz, quat_from_angle_axis,
    torch_rand_float,
)
from isaacgym import gymapi, gymtorch, gymutil

from legged_gym.envs.el_4090.spider_nomal.el_4090 import EL_4090
from legged_gym.utils.math_utils import quat_apply_yaw

from legged_gym.utils.LidarSensor.LidarSensor.lidar_sensor import LidarSensor
from legged_gym.utils.LidarSensor.LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


class EL_4090_Lidar(EL_4090):
    """EL_4090 + LiDAR perception with dual-GRU network (LidarPDActorCritic).

    Adds spherical LiDAR sensor, sector-safety reward computation, and
    LiDAR-aware observation assembly on top of EL_4090.

    Observation layout (before LidarWrapper):
      [0:66)   proprio (lin_vel, ang_vel, gravity, commands, dof_pos, dof_vel, actions)
      [66:N)   lidar_points_flat (num_lidar_points * 3)

    After LidarWrapper in runner: proprio + proximal*3 + distal_history*distal*3
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 task_name="el_4090_lidar"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         task_name=task_name)
        # Fixes #9: debug_viz in __init__, not _init_buffers
        self.debug_viz = getattr(cfg.env, "debug_viz", False)

    # ==================================================================
    # Buffer initialisation
    # Fixes #2: each method handles ONE concern
    # ==================================================================

    def _init_buffers(self):
        super()._init_buffers()
        self._lidar_done_this_step = False
        self._init_lidar_buffers()
        self._init_sector_buffers()
        self._init_lidar_sensor()
        # Fixes #8: init here, not lazily in reward function
        self.last_last_actions = torch.zeros_like(self.actions)

        # Auxiliary supervision buffer — height grid targets for compute_auxiliary_loss
        grid_x = self.cfg.terrain.measured_grid_x_count
        grid_y = self.cfg.terrain.measured_grid_y_count
        self.aux_obs_buf = torch.zeros(
            self.num_envs, grid_x * grid_y,
            device=self.device, dtype=torch.float)

    def _init_lidar_buffers(self):
        """Perception-only buffers (no task state mixed in)."""
        cfg = self.cfg.pd_risknet
        n_pts = int(cfg.num_lidar_points)
        d_max = float(cfg.ray_max_distance)

        self.lidar_points_base = torch.zeros(
            self.num_envs, n_pts, 3, device=self.device,
            dtype=torch.float, requires_grad=False,
        )
        self.raycast_distances = torch.full(
            (self.num_envs, n_pts), d_max, device=self.device,
            dtype=torch.float, requires_grad=False,
        )
        self._raw_distances = torch.full(
            (self.num_envs, n_pts), d_max, device=self.device,
            dtype=torch.float, requires_grad=False,
        )

    # ==================================================================
    # Sector safety buffers
    # ==================================================================

    def _init_sector_buffers(self):
        cd_cfg = self.cfg.cmd_safe
        n_sec = int(self.cfg.pd_risknet.n_sectors)

        # Precompute elliptical body radius per sector
        a = float(cd_cfg.body_semi_length)
        b = float(cd_cfg.body_semi_width)
        sec_size = 2.0 * math.pi / n_sec
        angles = torch.linspace(
            -math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size,
            n_sec, device=self.device,
        )
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        self._body_radius = a * b / torch.sqrt(
            b * b * cos_a * cos_a + a * a * sin_a * sin_a
        )  # (n_sec,)
        self._sector_centers = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1,
        )  # (n_sec, 2)

        # Runtime buffers
        self._sector_dists = torch.zeros(
            self.num_envs, n_sec, device=self.device, dtype=torch.float,
        )
        self._sector_safe = torch.zeros(
            self.num_envs, n_sec, device=self.device, dtype=torch.float,
        )
    # ==================================================================
    # LiDAR sensor
    # ==================================================================

    def _init_lidar_sensor(self):
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
        # Fixes #11: update_frequency from config, not hardcoded to 1/dt
        update_hz = float(getattr(ray_cfg, "update_frequency_hz",
                                  max(1.0, 1.0 / float(self.dt))))
        lidar_cfg = LidarConfig(
            sensor_type=LidarType.SIMPLE_GRID,
            dt=float(self.dt),
            update_frequency=update_hz,
            max_range=float(ray_cfg.max_distance),
            min_range=0.2,
            num_sensors=1,
            horizontal_line_num=int(ray_cfg.spherical_num_azimuth),
            vertical_line_num=int(ray_cfg.spherical_num_elevation),
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=float(getattr(ray_cfg, "vertical_fov_deg_min", -2.0)),
            vertical_fov_deg_max=float(getattr(ray_cfg, "vertical_fov_deg_max", 57.0)),
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
        """Build (vertices, triangles) for Warp raycasting.

        Fixes #5: extracted from _init_lidar_sensor.
        Fixes #11: plane_size from config, not hardcoded 100.0.
        """
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
                "EL_4090_Lidar requires trimesh terrain or plane mesh_type "
                "for LiDAR rendering."
            )
        return vertices, triangles_i32

    # ==================================================================
    # Height measurement grid
    # ==================================================================

    def _init_height_points(self):
        t = self.cfg.terrain
        if hasattr(t, "measured_grid_x_range"):
            x = torch.linspace(*t.measured_grid_x_range,
                               int(t.measured_grid_x_count), device=self.device)
            y = torch.linspace(*t.measured_grid_y_range,
                               int(t.measured_grid_y_count), device=self.device)
        else:
            x = torch.tensor(t.measured_points_x, device=self.device)
            y = torch.tensor(t.measured_points_y, device=self.device)

        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3,
                             device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        self._height_grid_x = x
        self._height_grid_y = y
        return points

    # ==================================================================
    # Noise — Fixes #6: dynamic indices from num_dof / num_actions
    # ==================================================================

    def _get_noise_scale_vec(self, cfg):
        n_dof = self.num_dof
        n_act = self.num_actions

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        i = 0
        noise_vec[i:i + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        i += 3
        noise_vec[i:i + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        i += 3
        noise_vec[i:i + 3] = noise_scales.gravity * noise_level
        i += 3
        noise_vec[i:i + 3] = 0.0  # commands
        i += 3
        noise_vec[i:i + n_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        i += n_dof
        noise_vec[i:i + n_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        i += n_dof
        noise_vec[i:i + n_act] = 0.0  # previous actions

        return noise_vec

    # ==================================================================
    # Proprio dimension — Fixes #7: computed from robot, not hardcoded
    # ==================================================================

    @property
    def _proprio_dim(self):
        return (3 + 3 + 3 + 3  # lin_vel, ang_vel, gravity, commands
                + self.num_dof + self.num_dof + self.num_actions)

    # ==================================================================
    # LiDAR update pipeline
    # ==================================================================

    def _update_lidar_history(self):
        if self.lidar_sensor is None:
            return

        # Update sensor pose in world frame
        self.sensor_quat_tensor.copy_(
            quat_mul(self.base_quat, self._sensor_offset_quat))
        self.sensor_pos_tensor.copy_(
            self.base_pos + quat_apply(self.base_quat, self._sensor_translation))

        lidar_points, lidar_dist = self.lidar_sensor.update()
        points_sensor = lidar_points.view(self.num_envs, -1, 3)
        n_points = points_sensor.shape[1]
        dist = lidar_dist.view(self.num_envs, -1)

        self._raw_distances.copy_(dist)

        # Transform: sensor frame → base frame
        quat_1x4 = self._sensor_offset_quat[0:1]
        n_total = int(points_sensor.numel() // 3)
        points_base = quat_apply(
            quat_1x4.expand(n_total, 4), points_sensor.reshape(-1, 3),
        ).reshape(self.num_envs, n_points, 3) + self._sensor_translation.unsqueeze(1)

        # Compute sector safety from clean points BEFORE domain randomization
        self._compute_sector_safety_impl(points_base, dist)

        # Apply domain randomization for network input
        points_base, dist = self._apply_lidar_domain_rand(points_base, dist)

        self.lidar_points_base.copy_(points_base)
        self.raycast_distances.copy_(dist)

        self._lidar_done_this_step = True

    def _apply_lidar_domain_rand(self, points_base, dist):
        """Point masking + distance noise for LiDAR domain randomization."""
        # Point masking
        mask_ratio = float(getattr(self.cfg.domain_rand,
                                   "lidar_point_mask_ratio", 0.0))
        if mask_ratio > 0.0:
            rand_mask = torch.rand_like(dist) < mask_ratio
            lo, hi = self.cfg.domain_rand.lidar_point_mask_value_range
            fake_dist = torch.rand_like(dist) * (hi - lo) + lo
            dir_norm = torch.linalg.norm(points_base, dim=-1, keepdim=True).clamp(min=1e-6)
            dir_unit = points_base / dir_norm
            points_base = torch.where(
                rand_mask.unsqueeze(-1),
                dir_unit * fake_dist.unsqueeze(-1),
                points_base,
            )
            dist = torch.where(rand_mask, fake_dist, dist)

        # Distance noise
        noise_ratio = float(getattr(self.cfg.domain_rand,
                                    "lidar_distance_noise_ratio", 0.0))
        if noise_ratio > 0.0:
            scale = 1.0 + (2.0 * torch.rand_like(dist) - 1.0) * noise_ratio
            points_base = points_base * scale.unsqueeze(-1)
            dist = dist * scale

        return points_base, dist

    # ==================================================================
    # Sector safety
    # ==================================================================


    def _compute_sector_safety_impl(self, pts: torch.Tensor, dist: torch.Tensor):
        cd_cfg = self.cfg.cmd_safe
        n_sec = int(self.cfg.pd_risknet.n_sectors)
        sec_size = 2.0 * math.pi / n_sec
        d_max = float(self.cfg.pd_risknet.ray_max_distance)

        n_points = pts.shape[1]

        # Ground filter: world-frame z ≈ 0
        pts_flat = pts.reshape(-1, 3)
        env_ids = torch.arange(self.num_envs, device=self.device).repeat_interleave(n_points)
        pts_world = (quat_apply(self.base_quat[env_ids], pts_flat)
                     .reshape(self.num_envs, n_points, 3)
                     + self.base_pos.unsqueeze(1))
        is_ground = pts_world[..., 2].abs() < 0.05

        # Overhead filter: body-frame z > threshold
        is_overhead = pts[..., 2] > cd_cfg.z_thresh_high

        z_mask = is_ground | is_overhead
        dist = torch.where(z_mask, torch.full_like(dist, d_max), dist)

        # Per-sector min distance
        body_azimuth = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor(
            (body_azimuth + math.pi) / sec_size
        ).long().clamp(min=0, max=n_sec - 1)

        min_dist = torch.full(
            (dist.shape[0], n_sec), 1e9, device=dist.device, dtype=dist.dtype,
        )
        min_dist.scatter_reduce_(1, sec_ids, dist, reduce='amin', include_self=False)

        # Body radius compensation → effective distance
        d_eff = torch.clamp(min_dist - self._body_radius.unsqueeze(0), min=0.0)
        self._sector_dists.copy_(d_eff)

        # Safety factor [0, 1]
        d_safety = float(cd_cfg.d_safety)
        d_safe_max = float(cd_cfg.d_safe_max)
        safe = torch.clamp((d_eff - d_safety) / (d_safe_max - d_safety), 0.0, 1.0)
        self._sector_safe.copy_(safe)

    # ==================================================================
    # Observations — Fixes #7: uses dynamic _proprio_dim
    # ==================================================================

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        # LiDAR data passed to wrap_obs via lidar_points_base parameter instead.

        # Auxiliary height supervision target — relative height grid (same format as base class)
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1, 1.
        ) * self.obs_scales.height_measurements
        self.aux_obs_buf[:] = heights

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ==================================================================
    # Step callback — EL_4090 uses post_physics_step (not _post_physics_step_callback)
    # ==================================================================

    def post_physics_step(self):
        super().post_physics_step()
        if not self._lidar_done_this_step:
            self._update_lidar_history()
        self._lidar_done_this_step = False

    # ==================================================================
    # Debug visualization
    # ==================================================================

    def _draw_debug_vis(self):
        """LiDAR point cloud: proximal (yellow), distal (red); velocity arrows; height grid boundary."""
        if self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        env_id = 0

        # ── Velocity arrows (from ElSpider, for all envs) ──
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        cmd_vel_world = quat_apply_yaw(self.base_quat, self.commands[:, :3]).cpu().numpy()
        cmd_vel_world[:, 2] = 0.0
        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))

        # ── LiDAR points for env 0 ──
        pts_base = self.lidar_points_base[env_id]
        if pts_base.abs().sum() == 0:
            return

        # Transform to sensor frame for theta computation (manual inverse rotation)
        sensor_q = self._sensor_offset_quat[env_id]
        pts_centered = pts_base - self._sensor_translation[env_id].unsqueeze(0)
        conj = sensor_q * torch.tensor([-1, -1, -1, 1], device=self.device)
        conj_vec = conj[:3].unsqueeze(0).expand(pts_centered.shape[0], 3)
        cross = 2.0 * torch.cross(conj_vec, pts_centered, dim=-1)
        pts_sensor = pts_centered + conj[3] * cross + torch.cross(conj_vec, cross, dim=-1)

        eps = 1e-8
        theta = torch.atan2(pts_sensor[:, 2], torch.linalg.norm(pts_sensor[:, :2], dim=1) + eps)
        split_rad = float(self.cfg.pd_risknet.split_theta_deg) * math.pi / 180.0
        prox_mask = theta >= split_rad
        dist_mask = ~prox_mask

        # Transform to world frame
        base_pos = self.base_pos[env_id].unsqueeze(0).repeat(pts_base.shape[0], 1)
        base_quat = self.base_quat[env_id].unsqueeze(0).repeat(pts_base.shape[0], 1)
        pts_world = base_pos + quat_apply(base_quat, pts_base)

        # Filter invalid (no-hit / sky) points
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        valid_mask = self.raycast_distances[env_id] < (d_max - 0.001)

        # Draw proximal (yellow)
        prox_draw = prox_mask & valid_mask
        prox_pts = pts_world[prox_draw].cpu().numpy()
        if len(prox_pts) > 0:
            prox_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for p in prox_pts:
                sphere_pose = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])), r=None)
                gymutil.draw_lines(prox_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        # Draw distal (red)
        dist_draw = dist_mask & valid_mask
        dist_pts = pts_world[dist_draw].cpu().numpy()
        if len(dist_pts) > 0:
            dist_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
            for p in dist_pts:
                sphere_pose = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])), r=None)
                gymutil.draw_lines(dist_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        # ── Height grid boundary (green rectangle in world XY plane) ──
        hx = self._height_grid_x
        hy = self._height_grid_y
        nx, ny = len(hx), len(hy)
        boundary = []
        boundary.extend([(hx[i].item(), hy[0].item(), 0.0) for i in range(nx)])
        boundary.extend([(hx[-1].item(), hy[j].item(), 0.0) for j in range(1, ny)])
        boundary.extend([(hx[i].item(), hy[-1].item(), 0.0) for i in range(nx - 2, -1, -1)])
        boundary.extend([(hx[0].item(), hy[j].item(), 0.0) for j in range(ny - 2, 0, -1)])
        b_pts = torch.tensor(boundary, device=self.device, dtype=torch.float)
        b_quat = self.base_quat[env_id].unsqueeze(0).expand(b_pts.shape[0], 4)
        b_pos = self.base_pos[env_id, :3].unsqueeze(0).expand(b_pts.shape[0], 3)
        b_world = quat_apply_yaw(b_quat, b_pts) + b_pos
        b_world[:, 2] = 0.0
        b_list = b_world.cpu().numpy().tolist()
        idx_bl = 0
        idx_br = nx - 1
        idx_tr = nx - 1 + ny - 1
        idx_tl = nx - 1 + ny - 1 + nx - 1
        corners = [b_list[idx_bl], b_list[idx_br], b_list[idx_tr], b_list[idx_tl]]
        for i in range(4):
            self.vis.draw_boldline(env_id, [corners[i], corners[(i + 1) % 4]],
                                   rad=0.01, resolution=6, color=(0, 1, 0))

    def draw_foot_hip_positions(self):
        """Suppressed: clear_lines here would wipe LiDAR debug viz drawn in _draw_debug_vis."""

    # ==================================================================
    # Reset — Fixes #5: _reset_lidar_buffers extracted, no duplication
    # ==================================================================

    def _reset_lidar_buffers(self, env_ids):
        """Reset LiDAR buffers for given env_ids (single source of truth)."""
        if len(env_ids) == 0:
            return
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = d_max
        self._raw_distances[env_ids] = d_max
        self._update_lidar_history()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self._reset_lidar_buffers(env_ids)

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
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
    # Commands — Fixes #11: deadzone from config, not hardcoded 0.2
    # ==================================================================

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)

        cmd_thresh = float(getattr(self.cfg.commands, "cmd_deadzone", 0.2))
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > cmd_thresh
        ).unsqueeze(1)

    # ==================================================================
    # Termination — Fixes #3: super().check_termination() first
    # ==================================================================

    def check_termination(self):
        super().check_termination()

        if getattr(self.cfg.env, "enable_fall_termination", False):
            g_thresh = float(getattr(self.cfg.env,
                                     "fall_projected_gravity_z_threshold", -0.1))
            h_thresh = float(getattr(self.cfg.env,
                                     "fall_base_height_threshold", 0.12))
            flipped = self.projected_gravity[:, 2] > g_thresh
            low_base = self.base_pos[:, 2] < h_thresh
            self.reset_buf |= (flipped | low_base)
            self.terminate_buf |= (flipped | low_base)

    # ==================================================================
    # Rewards
    # ==================================================================

    # _reward_termination, _reward_collision, _reward_feet_stumble,
    # _reward_dof_pos_limits: resolved via MRO to LeggedRobotRewMixin.
    #
    # _reward_ang_vel_yaw_penalty, _reward_curvature: removed (tracking
    # rewards already cover these signals).

    # -- Cmd-safe velocity rewards --

    def _reward_cmd_safe_vel(self):
        cd_cfg = self.cfg.cmd_safe
        cmd_2d = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_2d, dim=1, keepdim=True).clamp(min=1e-8)

        safe = self._sector_safe
        centers = self._sector_centers

        align = torch.clamp(torch.matmul(cmd_2d, centers.T), min=0.0)
        weighted = safe * align
        v_safe = torch.matmul(weighted, centers)
        v_safe_norm = torch.norm(v_safe, dim=1, keepdim=True).clamp(min=1e-8)
        v_safe = v_safe * torch.clamp(cmd_norm / v_safe_norm, max=1.0)

        v_actual = self.base_lin_vel[:, :2]
        vel_err = torch.sum(torch.square(v_actual - v_safe), dim=1)
        sigma = float(cd_cfg.cmd_safe_sigma)
        return torch.exp(-vel_err / sigma)

    def _reward_sector_dist_penalty(self):
        """Directionally-weighted exponential distance penalty.

        penalty_j = relu(exp(sigma*d_j) - exp(sigma*d_thresh)) / (1 - exp(sigma*d_thresh))
        weight_j   = cos²(θ_j/2) centred on commanded velocity direction
        r = -sum_j(w_j * p_j)
        """
        cd_cfg = self.cfg.cmd_safe
        d_thresh = float(cd_cfg.dist_penalty_thresh)
        sigma = float(getattr(cd_cfg, 'exp_sigma', -1.0))

        d = self._sector_dists  # (N, 36)

        # Exponential penalty: 1 at d=0, 0 at d=d_thresh, smooth in between
        exp_d = torch.exp(sigma * d)
        exp_thresh = math.exp(sigma * d_thresh)
        penalty = torch.clamp(
            (exp_d - exp_thresh) / max(1.0 - exp_thresh, 1e-8), min=0.0)

        # Directional weight: cos²(θ/2) = (1 + cos(θ)) / 2
        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
        cmd_dir = cmd_xy / cmd_norm.clamp(min=1e-8)
        zero_cmd = cmd_norm.squeeze(-1) < 0.01
        if zero_cmd.any():
            cmd_dir[zero_cmd] = torch.tensor([1.0, 0.0], device=self.device,
                                             dtype=cmd_dir.dtype)

        cos_theta = torch.matmul(cmd_dir, self._sector_centers.T)  # (N, 36)
        w = (1.0 + cos_theta) / 2.0

        return -(w * penalty).sum(dim=1)

    # _reward_action_rate2: second-order action smoothing.  Currently
    # scale=0 in config (no training signal); kept for future use.
    def _reward_action_rate2(self):
        rate2 = self.actions - 2.0 * self.last_actions + self.last_last_actions
        self.last_last_actions[:] = self.last_actions
        return torch.sum(torch.square(rate2), dim=1)
