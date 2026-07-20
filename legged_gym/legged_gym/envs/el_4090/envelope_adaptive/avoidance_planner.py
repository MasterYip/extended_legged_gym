"""LiDAR angular-distance curve + spline peak-finding + direction-weighted avoidance."""

import torch
import numpy as np
from scipy.interpolate import splrep, splev


def _quat_apply(q, v):
    """Apply quaternion rotation q (N,4) to vectors v (N,3)."""
    xyz = q[:, :3]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[:, 3:4] * t + torch.cross(xyz, t, dim=-1)


def _peak_prominence(d_smooth, peak_idx):
    """Peak prominence: how much the peak stands above the highest saddle
    to any higher peak on either side.  Scans outward along the shorter
    angular path (wrapping through 0 / n-1 is safe)."""
    n = len(d_smooth)
    peak_d = d_smooth[peak_idx]

    left_valley = peak_d
    for j in range(1, n):
        idx = (peak_idx - j) % n
        left_valley = min(left_valley, d_smooth[idx])
        if d_smooth[idx] > peak_d:
            break

    right_valley = peak_d
    for j in range(1, n):
        idx = (peak_idx + j) % n
        right_valley = min(right_valley, d_smooth[idx])
        if d_smooth[idx] > peak_d:
            break

    return peak_d - max(left_valley, right_valley)


def compute_safe_velocity(
    lidar_points_base,     # (E, N, 3) body-frame
    raycast_distances,     # (E, N)
    base_pos,              # (E, 3) world-frame
    base_quat,             # (E, 4) world-frame
    commands,              # (E, num_commands) — 仅读取 [:, 0:2] (vx, vy)
    cfg,                   # config.avoidance
    prev_theta=None,       # previous output theta for hysteresis (rad)
):
    E = lidar_points_base.shape[0]
    device = lidar_points_base.device

    ground_thresh = float(getattr(cfg, "ground_threshold", 0.05))
    min_valid = float(getattr(cfg, "min_valid_dist", 0.15))
    max_range = float(getattr(cfg, "max_valid_dist", 10.0))
    a_ell = float(getattr(cfg, "ellipse_a", 0.6))
    b_ell = float(getattr(cfg, "ellipse_b", 0.3))
    spline_s = float(getattr(cfg, "spline_smoothing", 0.5))
    cmd_bias = float(getattr(cfg, "cmd_bias", 0.3))
    cap_distance = float(getattr(cfg, "cap_distance", 2.0))
    n_azimuth = int(getattr(cfg, "n_azimuth", 40))

    safe_vx = commands[:, 0].clone()
    safe_vy = commands[:, 1].clone()
    debug = {}

    for env_id in range(E):
        vx_d = safe_vx[env_id].item()
        vy_d = safe_vy[env_id].item()
        cmd_speed = np.hypot(vx_d, vy_d)
        if cmd_speed < 1e-6:
            continue

        theta_cmd = np.arctan2(vy_d, vx_d)  # body-frame [-π, π]
        theta_cmd_02pi = theta_cmd if theta_cmd >= 0 else theta_cmd + 2*np.pi

        # ---- Step 1: preprocess (world-frame ground + noise filter) ----
        pts = lidar_points_base[env_id]          # (N, 3) body-frame
        dists = raycast_distances[env_id]         # (N,)
        bp = base_pos[env_id]                     # (3,)
        bq = base_quat[env_id]                    # (4,)

        pts_world = bp.unsqueeze(0) + _quat_apply(
            bq.unsqueeze(0).expand(pts.shape[0], 4), pts)
        is_ground = pts_world[:, 2].abs() < ground_thresh
        is_noise = dists < min_valid
        valid = ~(is_ground | is_noise)

        # ---- Step 2: angular-distance curve ----
        # LiDAR bin a maps to SENSOR azimuth π-2π·a/(N-1).
        # Sensor→body azimuth: θ_body = π - θ_sensor = 2π·a/(N-1) ∈ [0, 2π).
        n_elev = pts.shape[0] // n_azimuth
        pts_3d = pts.reshape(n_azimuth, n_elev, 3).transpose(0, 1).contiguous()
        dists_3d = dists.reshape(n_azimuth, n_elev).t().contiguous()
        valid_3d = valid.reshape(n_azimuth, n_elev).t().contiguous()

        d_curve = np.full(n_azimuth, cap_distance, dtype=np.float64)
        capped = np.ones(n_azimuth, dtype=bool)  # no-hit directions are all capped
        for a in range(n_azimuth):
            col_dists = dists_3d[:, a]
            col_valid = valid_3d[:, a]
            if col_valid.any():
                raw_d = col_dists[col_valid].min().item()
                d_curve[a] = min(raw_d, cap_distance)
                capped[a] = raw_d >= cap_distance
        # add bias only to capped (open) directions
        for a in range(n_azimuth):
            if capped[a]:
                body_theta = 2.0 * np.pi * a / (n_azimuth - 1)
                dtheta = body_theta - theta_cmd_02pi
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                d_curve[a] += cmd_bias * np.cos(dtheta)

        # ---- Step 3: spline + peak finding ----
        theta_arr = 2.0 * np.pi * np.arange(n_azimuth) / (n_azimuth - 1)  # [0, 2π]
        tck = splrep(theta_arr, d_curve, s=spline_s, per=True)
        d_smooth = splev(theta_arr, tck)

        peaks = []
        for i in range(n_azimuth):
            prev = (i - 1) % n_azimuth
            nxt = (i + 1) % n_azimuth
            if d_smooth[i] >= d_smooth[prev] and d_smooth[i] >= d_smooth[nxt]:
                thresh = d_smooth[i] * 0.75 + d_curve.min() * 0.25  # 3/4 height
                left = i
                for _ in range(n_azimuth):
                    cand = (left - 1) % n_azimuth
                    if d_smooth[cand] > thresh and cand != i:
                        left = cand
                    else:
                        break
                right = i
                for _ in range(n_azimuth):
                    cand = (right + 1) % n_azimuth
                    if d_smooth[cand] > thresh and cand != i:
                        right = cand
                    else:
                        break
                width = (right - left) % n_azimuth
                if width == 0:
                    width = n_azimuth
                fwhm = width * (2 * np.pi / n_azimuth)

                # ---- Step 4: quality ----
                theta_p = theta_arr[i]  # [0, 2π)
                theta_p_pi = theta_p if theta_p <= np.pi else theta_p - 2*np.pi
                d_ell = (a_ell * b_ell / np.sqrt(
                    (b_ell * np.cos(theta_p)) ** 2 +
                    (a_ell * np.sin(theta_p)) ** 2))
                prominence = _peak_prominence(d_smooth, i)
                quality = prominence * fwhm if d_smooth[i] > d_ell else 0.0

                # ---- Step 5: direction weighting ----
                dtheta = theta_p - theta_cmd_02pi
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                w = np.cos(dtheta / 2.0)
                score = quality * w

                if quality > 0:
                    peaks.append((score, theta_p_pi, dtheta, w, d_smooth[i]))

        # ---- Step 6-7: output ----
        if peaks:
            peaks.sort(key=lambda x: x[0], reverse=True)
            best = peaks[0]

            # hysteresis: prefer previous direction unless new best is significantly better
            if prev_theta is not None:
                # find peak closest to previous theta
                prev_match = min(peaks, key=lambda p: abs(
                    np.arctan2(np.sin(p[1] - prev_theta),
                               np.cos(p[1] - prev_theta))))
                if prev_match is not best:
                    hysteresis_ratio = 1.5
                    if best[0] < prev_match[0] * hysteresis_ratio:
                        best = prev_match

            _, theta_out, dtheta_best, w_best, _ = best
            out_speed = cmd_speed * w_best
            safe_vx[env_id] = out_speed * np.cos(theta_out)
            safe_vy[env_id] = out_speed * np.sin(theta_out)
        else:
            theta_out, w_best = None, None
            safe_vx[env_id] = 0.0
            safe_vy[env_id] = 0.0

        if env_id == 0:
            debug = {
                "theta": theta_arr,
                "d_raw": d_curve,
                "d_smooth": d_smooth,
                "peaks": [(p[1], p[4]) for p in peaks],
                "best_theta": theta_out,
                "theta_cmd": theta_cmd,
            }

    return safe_vx, safe_vy, debug
