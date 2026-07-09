"""包络参数计算模块。

当前返回上限值作为占位。后续实现：
- 输入 LiDAR 点云计算各方向障碍物距离
- 根据距离独立收缩 8 个包络参数
- 返回收缩后的参数用于绘制 + 避障速度计算
"""

import torch


def compute_envelope_params(
    lidar_points: torch.Tensor,      # (num_envs, N, 3) body-frame point cloud
    base_pos: torch.Tensor,           # (num_envs, 3) world-frame base position (reserved for future use)
    base_quat: torch.Tensor,          # (num_envs, 4) world-frame base orientation (reserved for future use)
    envelope_cfg,                     # config.envelope
) -> dict:
    """Compute 8 envelope parameters from LiDAR point cloud.

    Returns:
        dict with keys: x1, x3, l1, r1, l2, r2, l3, r3
        Each value is a (num_envs,) float tensor.
    """
    num_envs = lidar_points.shape[0]
    device = lidar_points.device

    return {
        "x1": torch.full((num_envs,), envelope_cfg.x1_max, device=device),
        "x3": torch.full((num_envs,), envelope_cfg.x3_max, device=device),
        "l1": torch.full((num_envs,), envelope_cfg.front_rear_max, device=device),
        "r1": torch.full((num_envs,), envelope_cfg.front_rear_max, device=device),
        "l2": torch.full((num_envs,), envelope_cfg.mid_max, device=device),
        "r2": torch.full((num_envs,), envelope_cfg.mid_max, device=device),
        "l3": torch.full((num_envs,), envelope_cfg.front_rear_max, device=device),
        "r3": torch.full((num_envs,), envelope_cfg.front_rear_max, device=device),
    }
