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

from legged_gym.envs.el_4090.envelope_adaptive.envelope_computer import compute_envelope_params


class EL_4090_EA(EL_4090):
    """EL_4090 with Envelope Adaptive obstacle avoidance.

    LiDAR point cloud processed externally -- no dual-GRU network.
    Policy receives proprioceptive observation only (66 dims).
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 task_name="el_4090_ea"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         task_name=task_name)

    # ==================================================================
    # Buffer initialisation
    # ==================================================================

    def _init_buffers(self):
        super()._init_buffers()
        self._init_lidar_buffers()
        self._init_lidar()

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
        # Envelope params buffer -- updated per step
        self.env_params = {
            k: torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            for k in ["x1", "x3", "l1", "r1", "l2", "r2", "l3", "r3"]
        }

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
        """Compute envelope params and draw the 3D prism."""
        if self.lidar_sensor is None:
            return

        params = compute_envelope_params(
            self.lidar_points_base,
            self.base_pos,
            self.base_quat,
            self.cfg.envelope,
        )
        for k, v in params.items():
            self.env_params[k].copy_(v)

        self._draw_envelope()

    def _draw_envelope(self):
        """Draw 3D prism envelope (top + bottom hexagons + 6 vertical edges)."""
        if self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)

        z_top = self.cfg.envelope.z_top
        z_bottom = self.cfg.envelope.z_bottom

        env_id = 0  # draw for env 0 only

        x1 = self.env_params["x1"][env_id].item()
        x3 = self.env_params["x3"][env_id].item()
        l1 = self.env_params["l1"][env_id].item()
        r1 = self.env_params["r1"][env_id].item()
        l2 = self.env_params["l2"][env_id].item()
        r2 = self.env_params["r2"][env_id].item()
        l3 = self.env_params["l3"][env_id].item()
        r3 = self.env_params["r3"][env_id].item()

        # 6 vertices in body frame (CCW from front-right)
        # x2 = 0 is implicit
        vertices_top = torch.tensor([
            [x1,  r1, z_top],   # 0: front-right
            [0.0, r2, z_top],   # 1: mid-right
            [x3,  r3, z_top],   # 2: rear-right
            [x3, -l3, z_top],   # 3: rear-left
            [0.0, -l2, z_top],  # 4: mid-left
            [x1, -l1, z_top],   # 5: front-left
        ], device=self.device, dtype=torch.float)

        vertices_bottom = torch.tensor([
            [x1,  r1, z_bottom],
            [0.0, r2, z_bottom],
            [x3,  r3, z_bottom],
            [x3, -l3, z_bottom],
            [0.0, -l2, z_bottom],
            [x1, -l1, z_bottom],
        ], device=self.device, dtype=torch.float)

        # Transform to world frame
        bp = self.base_pos[env_id]
        bq = self.base_quat[env_id]

        top_world = bp + quat_apply(bq, vertices_top)
        bottom_world = bp + quat_apply(bq, vertices_bottom)

        # Build line segments: top hexagon (6) + bottom hexagon (6) + verticals (6) = 18
        num_lines = 18
        verts_list = []
        color_list = []

        envelope_color = (0.0, 1.0, 0.5)  # green-cyan

        for hex_verts in [top_world, bottom_world]:
            for i in range(6):
                j = (i + 1) % 6
                verts_list.extend(hex_verts[i].cpu().numpy().tolist())
                verts_list.extend(hex_verts[j].cpu().numpy().tolist())
                color_list.extend(envelope_color)
                color_list.extend(envelope_color)

        # Vertical edges
        for i in range(6):
            verts_list.extend(top_world[i].cpu().numpy().tolist())
            verts_list.extend(bottom_world[i].cpu().numpy().tolist())
            color_list.extend(envelope_color)
            color_list.extend(envelope_color)

        verts_np = np.array(verts_list, dtype=np.float32)
        colors_np = np.array(color_list, dtype=np.float32)
        self.gym.add_lines(self.viewer, self.envs[env_id], num_lines,
                           verts_np, colors_np)

    # ==================================================================
    # Avoidance velocity (passthrough stub)
    # ==================================================================

    def _compute_avoidance_vel(self):
        """Compute obstacle-avoidance velocity from envelope params.

        Current: passthrough -- returns raw self.commands unchanged.
        Future: use self.env_params + self.commands to compute safe velocity.
        """
        pass  # self.commands is used as-is

    # ==================================================================
    # Step hooks
    # ==================================================================

    def post_physics_step(self):
        super().post_physics_step()
        self._update_lidar()
        self._update_envelope()
        self._compute_avoidance_vel()

    # ==================================================================
    # Reset
    # ==================================================================

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        d_max = float(self.cfg.raycaster.max_distance)
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = d_max

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
    # Debug viz suppressed (envelope drawn separately)
    # ==================================================================

    def draw_foot_hip_positions(self):
        """Suppressed: clear_lines in _draw_envelope replaces debug viz."""
        pass
    