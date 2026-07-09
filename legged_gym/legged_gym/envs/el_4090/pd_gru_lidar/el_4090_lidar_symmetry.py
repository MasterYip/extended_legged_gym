"""Left-right symmetry augmentation for EL_4090 LiDAR perception.

Pure-function module: no environment reads, no global state, no config
imports.  All parameters are explicit — callers bind sensor data and
dimension constants via ``functools.partial`` (see
:meth:`OnPolicyRunner._setup_symmetry`).

Observation layout (after ``LidarWrapper``)::

    [0:proprio_dim)                         proprioceptive
    [proprio_dim : proprio_dim + prox*3)    proximal LiDAR  (prox points)
    [proprio_dim + prox*3 : end)            distal LiDAR    (dist points x history)
"""

from __future__ import annotations

from typing import Tuple

import torch

from legged_gym.utils.pointcloud_geometry import sort_points_by_angular_key


@torch.no_grad()
def get_el4090_lidar_xsym_obs_act(
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    env=None,
    obs_type: str = "policy",
    *,
    proprio_dim: int = 66,
    proximal_points: int = 256,
    distal_history_points: int = 640,
    num_dof: int = 18,
    height_grid_x_count: int = 17,
    height_grid_y_count: int = 11,
    sensor_quat: torch.Tensor | None = None,
    sensor_trans: torch.Tensor | None = None,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """Left-right (Y-axis) symmetry for EL_4090 LiDAR observations and actions.

    Returns ``[batch * 2, dim]``: first half = original, second half = mirrored.

    Args:
        obs: Wrapped observation after ``LidarWrapper``, shape ``[B, D]``.
        actions: Joint position targets, shape ``[B, num_dof]``.
        env: Unused — kept for PPO caller compatibility.
        obs_type:
            ``"policy"`` — full proprioceptive + LiDAR symmetry.
            ``"critic"`` — same as policy (shared buffer in LiDAR mode).
            ``"auxiliary"`` — height-grid Y-flip only (reserved, not called by PPO).
        proprio_dim: Dimension of the proprioceptive prefix (default 66).
        proximal_points: Number of proximal LiDAR points (default 256).
        distal_history_points: Total distal LiDAR points across history (default 640).
        num_dof: Degrees-of-freedom = leg count x joints per leg (default 18).
        height_grid_x_count, height_grid_y_count: Grid dimensions for auxiliary obs.
        sensor_quat: Sensor offset quaternion ``[1, 4]``; ``None`` -> identity.
        sensor_trans: Sensor translation ``[1, 3]``; ``None`` -> zeros.

    Returns:
        ``(augmented_obs, augmented_actions)`` — each is ``None`` if the
        corresponding input was ``None``.
    """
    # ---- resolve sensor frame transform ----------------------------------
    _device = obs.device if obs is not None else actions.device
    if sensor_quat is None:
        sensor_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=_device)
    if sensor_trans is None:
        sensor_trans = torch.zeros(1, 3, device=_device)

    # ---- observations ----------------------------------------------------
    if obs is not None:
        # --- dimension guard ---
        prox_len = proximal_points * 3
        dist_len = distal_history_points * 3
        if obs_type in ("policy", "critic"):
            expected_len = proprio_dim + prox_len + dist_len
            if obs.shape[-1] != expected_len:
                raise ValueError(
                    f"[SYMMETRY] Unexpected obs dim {obs.shape[-1]}, "
                    f"expected {expected_len} (proprio={proprio_dim}"
                    f" + prox={prox_len} + distal={dist_len})"
                )
        elif obs_type == "auxiliary":
            expected_len = height_grid_x_count * height_grid_y_count
            if obs.shape[-1] != expected_len:
                raise ValueError(
                    f"[SYMMETRY] Unexpected aux obs dim {obs.shape[-1]}, "
                    f"expected {expected_len}"
                    f" (grid {height_grid_x_count}x{height_grid_y_count})"
                )
        else:
            raise ValueError(
                f"[SYMMETRY] Unknown obs_type: '{obs_type}'. "
                f"Expected one of: 'policy', 'critic', 'auxiliary'."
            )

        # --- auxiliary path: height grid Y-flip only (no clone needed) ---
        if obs_type == "auxiliary":
            obs_aug = torch.cat([obs, torch.flip(
                obs.reshape(-1, height_grid_x_count, height_grid_y_count),
                dims=[2],
            ).reshape_as(obs)], dim=0)
            return obs_aug, _process_actions(actions, num_dof)

        # cat first: obs_aug[B:] starts as a copy of obs, then modify in-place
        obs_aug = torch.cat([obs, obs], dim=0)

        B = obs.shape[0]

        # --- proprioceptive sign flips (read from [:B], write negated to [B:]) ---
        obs_aug[B:, 1] = -obs_aug[:B, 1]   # lin_vel y
        obs_aug[B:, 3] = -obs_aug[:B, 3]   # ang_vel x
        obs_aug[B:, 5] = -obs_aug[:B, 5]   # ang_vel z
        obs_aug[B:, 7] = -obs_aug[:B, 7]   # projected_gravity y
        obs_aug[B:, 10] = -obs_aug[:B, 10] # command lin_vel y
        obs_aug[B:, 11] = -obs_aug[:B, 11] # command ang_vel yaw

        # --- joint swap left<->right (read from [:B], write to [B:]) ---
        half_dof = num_dof // 2
        dof_start = 12
        # dof_pos
        dof_pos_end = dof_start + num_dof
        obs_aug[B:, dof_start:dof_start + half_dof] = obs_aug[:B, dof_start + half_dof:dof_pos_end]
        obs_aug[B:, dof_start + half_dof:dof_pos_end] = obs_aug[:B, dof_start:dof_start + half_dof]

        # dof_vel
        vel_start = dof_pos_end
        vel_end = vel_start + num_dof
        obs_aug[B:, vel_start:vel_start + half_dof] = obs_aug[:B, vel_start + half_dof:vel_end]
        obs_aug[B:, vel_start + half_dof:vel_end] = obs_aug[:B, vel_start:vel_start + half_dof]

        # prev_actions
        act_start = vel_end
        act_end = act_start + num_dof
        obs_aug[B:, act_start:act_start + half_dof] = obs_aug[:B, act_start + half_dof:act_end]
        obs_aug[B:, act_start + half_dof:act_end] = obs_aug[:B, act_start:act_start + half_dof]

        # --- proximal LiDAR: Y-flip on [B:] view, then sort, write back ---
        prox_start = proprio_dim
        prox_end = prox_start + prox_len
        prox_flat = obs_aug[B:, prox_start:prox_end]
        prox_pts = prox_flat.reshape(-1, proximal_points, 3)
        prox_pts[:, :, 1] = -prox_pts[:, :, 1]
        prox_sorted = sort_points_by_angular_key(prox_pts, sensor_quat, sensor_trans)
        obs_aug[B:, prox_start:prox_end] = prox_sorted.reshape_as(prox_flat)

        # --- distal LiDAR: Y-flip on [B:] view, then sort, write back ---
        dist_flat = obs_aug[B:, prox_end:]
        dist_pts = dist_flat.reshape(-1, distal_history_points, 3)
        dist_pts[:, :, 1] = -dist_pts[:, :, 1]
        dist_sorted = sort_points_by_angular_key(dist_pts, sensor_quat, sensor_trans)
        obs_aug[B:, prox_end:] = dist_sorted.reshape_as(dist_flat)
    else:
        obs_aug = None

    # ---- actions ----------------------------------------------------------
    actions_aug = _process_actions(actions, num_dof)

    return obs_aug, actions_aug


def _process_actions(
    actions: torch.Tensor | None,
    num_dof: int,
) -> torch.Tensor | None:
    """Swap left <-> right leg action groups.  No sign negation (EL_4090 convention)."""
    if actions is None:
        return None
    half_dof = num_dof // 2
    act_m = actions.clone()
    act_m[:, :half_dof] = actions[:, half_dof:]
    act_m[:, half_dof:] = actions[:, :half_dof]
    return torch.cat([actions, act_m], dim=0)
