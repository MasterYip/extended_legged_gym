from __future__ import annotations

import math
import torch


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., :3]
    q_scalar = q[..., 3:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)

    out = v.clone()
    out.addcmul_(q_scalar, t)

    cross_buf = torch.empty_like(t)
    torch.cross(q_vec, t, dim=-1, out=cross_buf)
    out.add_(cross_buf)

    return out


class LidarWrapper:
    def __init__(
        self,
        num_envs: int,
        num_lidar_points: int,
        distal_history_length: int,
        proximal_points: int,
        distal_points: int,
        phi_threshold_deg: float,
        proprio_dim: int,
        device: torch.device,
        sensor_offset_quat: torch.Tensor | None = None,
        sensor_translation: torch.Tensor | None = None,
    ):
        self.num_envs = num_envs
        self.num_lidar_points = num_lidar_points
        self.distal_history_length = distal_history_length
        self.proximal_points = proximal_points
        self.distal_points = distal_points
        self.phi_threshold_rad = math.radians(phi_threshold_deg)
        self.proprio_dim = proprio_dim
        self.device = device

        if sensor_offset_quat is not None:
            self._sensor_conj = _quat_conjugate(sensor_offset_quat[0:1]).to(device)
        else:
            self._sensor_conj = None
        if sensor_translation is not None:
            self._sensor_t = sensor_translation[0:1].to(device)
        else:
            self._sensor_t = torch.zeros(1, 3, device=device)

        self._distal_window = torch.zeros(
            num_envs, distal_history_length, distal_points, 3,
            device=device, dtype=torch.float,
        )
        self._distal_frame_count = torch.zeros(num_envs, device=device, dtype=torch.long)

    @property
    def wrapped_obs_dim(self) -> int:
        return self.proprio_dim + self.proximal_points * 3 + self.distal_history_length * self.distal_points * 3

    def _cart_to_sphere(self, points_sensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, z = points_sensor[..., 0], points_sensor[..., 1], points_sensor[..., 2]
        r = torch.norm(points_sensor, dim=-1)
        azimuth = torch.atan2(y, x)
        phi = torch.asin(z / (r + 1e-9))
        return r, azimuth, phi

    def _to_sensor_frame(self, points_base: torch.Tensor) -> torch.Tensor:
        if self._sensor_conj is None and (self._sensor_t == 0).all():
            return points_base
        t = self._sensor_t.to(points_base.device)
        pts = points_base - t.unsqueeze(1)
        if self._sensor_conj is not None:
            q = self._sensor_conj.to(points_base.device)
            B, N = pts.shape[:2]
            pts = _quat_apply(q.expand(B * N, 4), pts.reshape(-1, 3)).reshape(B, N, 3)
        return pts

    def _from_sensor_frame(self, points_sensor: torch.Tensor) -> torch.Tensor:
        """sensor frame -> base frame 逆变换。

        _to_sensor_frame 的正向变换为: pts_sensor = conj * (pts_base - t)。
        逆变换: pts_base = conj⁻¹ * pts_sensor + t。
        """
        if self._sensor_conj is not None:
            q = _quat_conjugate(self._sensor_conj).to(points_sensor.device)
            B, N = points_sensor.shape[:2]
            pts = _quat_apply(q.expand(B * N, 4), points_sensor.reshape(-1, 3)).reshape(B, N, 3)
        else:
            pts = points_sensor
        t = self._sensor_t.to(points_sensor.device)
        if (t == 0).all():
            return pts
        return pts + t.unsqueeze(1)

    def _sort_by_angular_key(self, points: torch.Tensor) -> torch.Tensor:
        pts = self._to_sensor_frame(points)
        _, azimuth, phi = self._cart_to_sphere(pts)
        key = phi * (2.0 * math.pi) + azimuth
        order = torch.argsort(key, dim=1)
        return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))

    def _batch_fps(self, points: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        B, N_max, _ = points.shape
        device = points.device

        counts = mask.sum(dim=1)
        if (counts == 0).all():
            return torch.zeros(B, k, 3, device=device)

        M = int(counts.max().item())

        cumsum = mask.cumsum(dim=1) - 1
        positions = torch.where(mask, cumsum, torch.zeros_like(cumsum))
        padded = torch.zeros(B, M, 3, device=device)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N_max)
        valid_batch = batch_idx[mask]
        valid_pos = positions[mask]
        valid_pts = points[mask]
        padded[valid_batch, valid_pos] = valid_pts

        fill_mask = torch.arange(M, device=device).unsqueeze(0) >= counts.unsqueeze(1)
        if fill_mask.any():
            padded[fill_mask] = padded[:, 0:1, :].expand(-1, M, -1)[fill_mask]

        k_eff = min(k, M)

        selected = torch.zeros(B, k_eff, 3, device=device)
        dists = torch.full((B, M), float("inf"), device=device)
        farthest = torch.argmax(padded.norm(dim=-1), dim=1)

        diff_buf = torch.empty(B, M, 3, device=device)
        d_buf = torch.empty(B, M, device=device)

        for i in range(k_eff):
            selected[:, i] = padded[torch.arange(B, device=device), farthest]
            centroid = selected[:, i:i + 1, :]
            torch.sub(padded, centroid, out=diff_buf)
            diff_buf.square_()
            torch.sum(diff_buf, dim=-1, out=d_buf)
            torch.minimum(dists, d_buf, out=dists)
            farthest = torch.argmax(dists, dim=1)

        if k_eff < k:
            out = torch.zeros(B, k, 3, device=device)
            out[:, :k_eff] = selected
            return out
        return selected

    def _downsample_distal(self, points_sensor: torch.Tensor, mask: torch.Tensor, k: int,
                           azimuth: torch.Tensor | None = None,
                           phi: torch.Tensor | None = None) -> torch.Tensor:
        B, N = points_sensor.shape[:2]
        device = points_sensor.device

        if azimuth is None or phi is None:
            _, azimuth, phi = self._cart_to_sphere(points_sensor)

        sort_key = torch.where(
            mask,
            phi * (2.0 * math.pi) + azimuth,
            torch.full_like(phi, float("inf")),
        )
        sorted_idx = torch.argsort(sort_key, dim=1)

        counts = mask.sum(dim=1).clamp(min=1)
        k_eff = min(k, N)

        pos = torch.arange(k_eff, device=device).unsqueeze(0).expand(B, -1)
        denom = max(k_eff - 1, 1)
        uniform_pos = torch.round(
            pos.float() * (counts.unsqueeze(1).float() - 1.0) / float(denom)
        ).long()
        uniform_pos = torch.minimum(uniform_pos, (counts - 1).unsqueeze(1))

        orig_indices = torch.gather(sorted_idx, 1, uniform_pos)
        selected = torch.gather(points_sensor, 1, orig_indices.unsqueeze(-1).expand(-1, -1, 3))

        if k_eff < k:
            out = torch.zeros(B, k, 3, device=device)
            out[:, :k_eff] = selected
            keep = pos < counts.unsqueeze(1)
            out = out * keep.unsqueeze(-1)
            return out
        return selected

    def wrap_obs(
        self,
        obs_buf: torch.Tensor,
        lidar_points_base: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        B = obs_buf.shape[0]

        proprio = obs_buf[:, :self.proprio_dim]
        lidar_raw = obs_buf[:, self.proprio_dim:].reshape(B, -1, 3)

        pts_sensor = self._to_sensor_frame(lidar_raw)
        _, azimuth, phi = self._cart_to_sphere(pts_sensor)

        valid_mask = lidar_raw.abs().sum(dim=-1) > 0
        proximal_mask = (phi >= self.phi_threshold_rad) & valid_mask
        distal_mask = (phi < self.phi_threshold_rad) & valid_mask

        prox_k = self.proximal_points
        prox_fps = self._batch_fps(lidar_raw, proximal_mask, prox_k)
        prox_sorted = self._sort_by_angular_key(prox_fps)

        dist_k = self.distal_points
        dist_down = self._downsample_distal(pts_sensor, distal_mask, dist_k,
                                            azimuth=azimuth, phi=phi)
        dist_down = self._from_sensor_frame(dist_down)
        dist_sorted = self._sort_by_angular_key(dist_down)

        write_idx = (self._distal_frame_count % self.distal_history_length).long()
        self._distal_window[torch.arange(B, device=self.device), write_idx] = dist_sorted
        self._distal_frame_count += 1

        fill_mask = self._distal_frame_count < self.distal_history_length
        if fill_mask.any():
            fill_envs = fill_mask.nonzero(as_tuple=False).squeeze(-1)
            first_frame = self._distal_window[fill_envs, 0:1, :, :]
            self._distal_window[fill_envs] = first_frame.expand(-1, self.distal_history_length, -1, -1)

        distal_cat = self._distal_window.reshape(B, self.distal_history_length * dist_k, 3)
        distal_sorted = self._sort_by_angular_key(distal_cat)

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._distal_window[done_ids] = 0.0
            self._distal_frame_count[done_ids] = 0

        wrapped = torch.cat([
            proprio,
            prox_sorted.reshape(B, -1),
            distal_sorted.reshape(B, -1),
        ], dim=-1)

        return wrapped
