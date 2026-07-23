"""Fan-sector point cloud detection + step-based symmetric envelope shrink/grow.

包络状态即 condition 前 5 维(front_width/middle_width/back_width/forward_limit/backward_limit),
左右共享宽度值;收缩边界 = cfg.commands.ranges(单一事实来源)。
"""

import torch

# 包络 5 参数名(与 cfg.commands.condition_names 前 5 项一致)
ENVELOPE_PARAM_NAMES = (
    "front_width", "middle_width", "back_width", "forward_limit", "backward_limit",
)


def _cross2d(a, b):
    """2D cross product: a.x*b.y - a.y*b.x.  a,b: (..., 2)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _build_hex_edges(x1, x3, front_w, mid_w, back_w):
    """Return CCW inner hexagon 6 vertices (E, 6, 2), left-right symmetric.

    B=(x1,front_w) D=(0,mid_w) F=(x3,back_w) E=(x3,-back_w) C=(0,-mid_w) A=(x1,-front_w)
    """
    z = torch.zeros_like(x1)
    B = torch.stack([x1,  front_w], dim=-1)
    D = torch.stack([z,   mid_w], dim=-1)
    F = torch.stack([x3,  back_w], dim=-1)
    E_v = torch.stack([x3, -back_w], dim=-1)
    C = torch.stack([z,   -mid_w], dim=-1)
    A = torch.stack([x1, -front_w], dim=-1)
    return torch.stack([B, D, F, E_v, C, A], dim=1)  # (E, 6, 2)


def _offset_hexagon(verts, margin):
    """Radial outward offset, returns outer vertices (E, 6, 2)."""
    r = torch.norm(verts, dim=-1, keepdim=True).clamp(min=1e-8)
    return verts * (1.0 + margin / r)  # push each vertex outward along radial direction


def _point_inside_convex(pts, verts):
    """Check if points are inside CCW convex polygon. pts:(E,N,2|3) verts:(E,V,2) -> (E,N) bool."""
    pts_2d = pts[..., :2]                             # (E, N, 2)
    V = verts.shape[1]
    nxt = (torch.arange(V) + 1) % V
    edges = verts[:, nxt, :] - verts                  # (E, V, 2)
    rel = pts_2d.unsqueeze(2) - verts.unsqueeze(1)    # (E, N, V, 2)
    cross = _cross2d(edges.unsqueeze(1), rel)          # (E, N, V)
    return (cross >= 0).all(dim=-1)  # CCW -> all >= 0 = inside


def _sector_angles(verts):
    """Compute sector angular bounds. Returns dict: param_name -> [(start, end), ...].

    几何上仍是 8 个扇区;左右两侧扇区合并映射到同一对称参数(hit 取区间并集)。
    注意 _angle_in_range 的批量整体分支依赖几何不变量:backward_limit 扇区恒跨 ±π,
    其余扇区恒不跨(前扇区跨 0 但 start<end 数值成立)。
    """
    B, D, F, E_v, C, A = [verts[:, i, :] for i in range(6)]

    def _ang(p):
        return torch.atan2(p[..., 1], p[..., 0])

    I1 = A + (B - A) / 8.0          # AB 边靠 A 侧 1/8
    I7 = A + (B - A) * 7.0 / 8.0   # AB 边靠 B 侧 7/8
    G = (A + C) / 2.0               # AC 中点
    H = (C + E_v) / 2.0             # CE 中点
    J1 = E_v + (F - E_v) / 8.0      # EF 边靠 E 侧 1/8
    J7 = E_v + (F - E_v) * 7.0 / 8.0  # EF 边靠 F 侧 7/8
    K = (B + D) / 2.0               # BD 中点
    L = (D + F) / 2.0               # DF 中点

    a_A, a_B = _ang(A), _ang(B)
    a_E, a_F = _ang(E_v), _ang(F)
    a_I1, a_I7 = _ang(I1), _ang(I7)
    a_C, a_D = _ang(C), _ang(D)
    a_G, a_K = _ang(G), _ang(K)
    a_H, a_L = _ang(H), _ang(L)
    a_J1, a_J7 = _ang(J1), _ang(J7)

    return {
        "forward_limit":  [(a_A, a_B)],                # 前向全宽扇区 (跨 0)
        "backward_limit": [(a_F, a_E)],                # 后向全宽扇区 (跨 ±π)
        "front_width":    [(a_I7, a_D), (a_C, a_I1)],  # +y 侧 与 -y 侧 (扩展到侧边顶点D/C)
        "middle_width":   [(a_K, a_L), (a_H, a_G)],
        "back_width":     [(a_D, a_J7), (a_J1, a_C)],  # 扩展到侧边顶点D/C (注意: D→J7, J1→C 非wrap方向)
    }


def _angle_in_range(angles, start, end):
    """Check if angles fall in [start, end] range (handles 0/+/-pi wrapping)."""
    if (start <= end).all():
        return (angles >= start) & (angles <= end)
    else:
        return (angles >= start) | (angles <= end)


def compute_envelope_params(
    lidar_points,        # (E, N, 3) body-frame
    base_pos,            # (E, 3)  reserved
    base_quat,           # (E, 4)  reserved
    cfg,                 # 完整 env config (El4090EACfg) — 需 envelope 与 commands.ranges
    prev_params,         # dict of (E,) tensors, keys = ENVELOPE_PARAM_NAMES
    cooldown,            # (E, 5) int tensor — consecutive no-hit frames per parameter
    min_hex,             # (1, 6, 2) — pre-built minimum hexagon (on same device as pts)
):
    device = lidar_points.device
    ecfg = cfg.envelope

    margin = float(getattr(ecfg, "margin_distance", 0.5))
    shrink = float(getattr(ecfg, "shrink_step", 0.1))
    grow = float(getattr(ecfg, "grow_step", 0.02))

    # 收缩边界 = 底层 condition ranges(单一事实来源);数值天然 low<high,统一 clamp
    ranges = cfg.commands.ranges
    bounds = {name: getattr(ranges, name) for name in ENVELOPE_PARAM_NAMES}

    params = {k: prev_params[k].clone() for k in ENVELOPE_PARAM_NAMES}

    # Pre-create clamp bounds outside loop to avoid per-iteration tensor creation
    lows_t = {k: torch.tensor(v[0], device=device) for k, v in bounds.items()}
    highs_t = {k: torch.tensor(v[1], device=device) for k, v in bounds.items()}

    pts = lidar_points               # (E, N, 3) — body-frame, XY 判多边形, Z 判棱柱高度
    angles = torch.atan2(pts[..., 1], pts[..., 0])  # (E, N)  [-pi, pi]

    inner = _build_hex_edges(
        params["forward_limit"], params["backward_limit"],
        params["front_width"], params["middle_width"], params["back_width"],
    )
    outer = _offset_hexagon(inner, margin)
    hold_m = float(getattr(ecfg, "hold_margin", 0.1))
    inner_shrink = _offset_hexagon(inner, hold_m)  # shrink zone upper boundary

    sectors = _sector_angles(inner)

    outside_min = ~_point_inside_convex(pts, min_hex)   # (E, N) — fixed lower bound
    inside_outer = _point_inside_convex(pts, outer)       # (E, N)

    # Z-dimension filter: points must be within the 3D prism height
    z_bottom = float(ecfg.z_bottom)
    z_top = float(ecfg.z_top)
    pts_z = pts[..., 2]
    in_zone_z = (pts_z >= z_bottom) & (pts_z <= z_top)

    # Three-zone detection: shrink | hold | clear
    in_shrink = outside_min & _point_inside_convex(pts, inner_shrink) & in_zone_z
    in_hold   = ~_point_inside_convex(pts, inner_shrink) & inside_outer & in_zone_z

    threshold = int(getattr(cfg.envelope, "grow_cooldown_frames", 5))
    param_idx = {n: i for i, n in enumerate(ENVELOPE_PARAM_NAMES)}

    for name, intervals in sectors.items():
        idx = param_idx[name]
        hit_shrink = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
        hit_hold   = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
        for s_start, s_end in intervals:
            in_sector = _angle_in_range(angles, s_start.unsqueeze(-1), s_end.unsqueeze(-1))
            hit_shrink |= (in_shrink & in_sector).any(dim=-1)
            hit_hold   |= (in_hold & in_sector).any(dim=-1)

        # shrink resets cooldown; hold preserves; clear increments
        cooldown[:, idx] = torch.where(hit_shrink,
            torch.zeros_like(cooldown[:, idx]),
            torch.where(hit_hold, cooldown[:, idx], cooldown[:, idx] + 1),
        )

        should_shrink = hit_shrink
        should_grow   = ~hit_shrink & ~hit_hold & (cooldown[:, idx] >= threshold)

        if name == "backward_limit":
            delta = torch.where(should_shrink, shrink, torch.where(should_grow, -grow, 0.0))
        else:
            delta = torch.where(should_shrink, -shrink, torch.where(should_grow, grow, 0.0))
        params[name] = torch.clamp(params[name] + delta, lows_t[name], highs_t[name])

    return params, cooldown


def envelope_params_to_condition(params, cfg):
    """Convert 5 symmetric envelope parameters to 8-dim condition tensor.

    参数已在 ranges 内 clamp;此处保留防御性 clip(观测直接消费原始值,最后一道防线),
    再补算 3 个 directional_ratio 形态先验。

    Args:
        params: dict of (E,) tensors, keys = ENVELOPE_PARAM_NAMES
        cfg: full env config (El4090EACfg instance)

    Returns:
        (E, 8) tensor ordered as cfg.commands.condition_names
    """
    ranges = cfg.commands.ranges

    # -- 1. 防御性 clip 到 command_ranges --
    front_width = _clip_range(params["front_width"], *ranges.front_width)
    middle_width = _clip_range(params["middle_width"], *ranges.middle_width)
    back_width = _clip_range(params["back_width"], *ranges.back_width)
    forward_limit = _clip_range(params["forward_limit"], *ranges.forward_limit)
    backward_limit = _clip_range(params["backward_limit"], *ranges.backward_limit)

    # -- 2. 归一化 [0, 1] --
    fw_norm = _norm(front_width, *ranges.front_width)
    mw_norm = _norm(middle_width, *ranges.middle_width)
    bw_norm = _norm(back_width, *ranges.back_width)
    fwd_norm = _norm(forward_limit, *ranges.forward_limit)
    # backward_limit: flip sign first, then normalize
    bwd_norm = _norm(-backward_limit,
                     -ranges.backward_limit[1],  # -(-0.6) = 0.6
                     -ranges.backward_limit[0])  # -(-0.9) = 0.9

    # -- 3. directional_ratio 先验 --
    weights = cfg.commands.morphology_prior_weights
    eps = 1e-6
    # front
    fl = float(weights["front"].get("lateral", 0.5))
    flong = float(weights["front"].get("longitudinal", 0.5))
    front_prior = (flong * fwd_norm) / torch.clamp(flong * fwd_norm + fl * fw_norm, min=eps)
    # middle
    ml = float(weights["middle"].get("lateral", 1.0))
    middle_prior = 1.0 - mw_norm
    if ml <= 0.0:
        middle_prior = torch.full_like(mw_norm, 0.5)
    fw = min(max(float(getattr(cfg.commands, "morphology_middle_front_follow_weight", 0.0)), 0.0), 1.0)
    middle_prior = fw * front_prior + (1.0 - fw) * middle_prior
    # back
    bl = float(weights["back"].get("lateral", 0.5))
    blong = float(weights["back"].get("longitudinal", 0.5))
    back_prior = (blong * bwd_norm) / torch.clamp(blong * bwd_norm + bl * bw_norm, min=eps)

    # -- 4. Clamp priors --
    front_prior = torch.clamp(front_prior, 0.0, 1.0)
    middle_prior = torch.clamp(middle_prior, 0.0, 1.0)
    back_prior = torch.clamp(back_prior, 0.0, 1.0)

    # -- 5. 按 condition_names 顺序输出 --
    return torch.stack([
        front_width, middle_width, back_width,
        forward_limit, backward_limit,
        front_prior, middle_prior, back_prior,
    ], dim=-1)  # (E, 8)


def _make_min_hex(cfg, device='cpu'):
    """Build the minimum-extent hexagon (all params at shrink end). (1, 6, 2).

    backward_limit[1]=-0.6 is the shrunk end (closest backward = numerically larger).
    Must be on same device as lidar_points to avoid CPU/GPU mismatch in _point_inside_convex.
    """
    ranges = cfg.commands.ranges
    return _build_hex_edges(
        torch.tensor([ranges.forward_limit[0]], device=device),    # 0.6 (closest forward)
        torch.tensor([ranges.backward_limit[1]], device=device),   # -0.6 (closest backward)
        torch.tensor([ranges.front_width[0]], device=device),       # 0.3
        torch.tensor([ranges.middle_width[0]], device=device),      # 0.3
        torch.tensor([ranges.back_width[0]], device=device),        # 0.3
    )


def _clip_range(tensor, low, high):
    return torch.clamp(tensor, min=low, max=high)


def _norm(tensor, low, high):
    return torch.clamp((tensor - low) / max(high - low, 1e-6), 0.0, 1.0)
