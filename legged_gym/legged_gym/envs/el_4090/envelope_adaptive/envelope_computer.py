"""Fan-sector point cloud detection + step-based envelope shrink/grow."""

import torch



def _cross2d(a, b):
    """2D cross product: a.x*b.y - a.y*b.x.  a,b: (..., 2)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _build_hex_edges(x1, x3, l1, l2, l3, r1, r2, r3):
    """Return CCW inner hexagon 6 vertices (E, 6, 2).  Order: B->D->F->E->C->A.

    A=(x1,-l1) B=(x1,r1) C=(0,-l2) D=(0,r2) E=(x3,-l3) F=(x3,r3)
    """
    E = x1.shape[0]
    z = torch.zeros_like(x1)
    B = torch.stack([x1,  r1], dim=-1)
    D = torch.stack([z,    r2], dim=-1)
    F = torch.stack([x3,  r3], dim=-1)
    E_v = torch.stack([x3, -l3], dim=-1)
    C = torch.stack([z,   -l2], dim=-1)
    A = torch.stack([x1, -l1], dim=-1)
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
    """Compute 8 sector angular bounds (radians). Returns dict: param_name -> (start, end)."""
    B, D, F, E_v, C, A = [verts[:, i, :] for i in range(6)]

    def _ang(p):
        return torch.atan2(p[..., 1], p[..., 0])

    I1 = A + (B - A) / 8.0          # AB left 1/8
    I7 = A + (B - A) * 7.0 / 8.0   # AB right 7/8
    G = (A + C) / 2.0               # AC midpoint
    H = (C + E_v) / 2.0             # CE midpoint
    J1 = E_v + (F - E_v) / 8.0      # EF left 1/8
    J7 = E_v + (F - E_v) * 7.0 / 8.0  # EF right 7/8
    K = (B + D) / 2.0               # BD midpoint
    L = (D + F) / 2.0               # DF midpoint

    a_A, a_B = _ang(A), _ang(B)
    a_E, a_F = _ang(E_v), _ang(F)
    a_I1, a_I7 = _ang(I1), _ang(I7)
    a_G, a_K = _ang(G), _ang(K)
    a_H, a_L = _ang(H), _ang(L)
    a_J1, a_J7 = _ang(J1), _ang(J7)

    return {
        "x1": (a_A, a_B),          # front full-width (A->B CCW, crosses 0)
        "x3": (a_F, a_E),          # rear full-width  (F->E CCW, crosses +/-pi)
        "r1": (a_I7, a_K),
        "r2": (a_K, a_L),
        "r3": (a_L, a_J7),
        "l1": (a_G, a_I1),
        "l2": (a_H, a_G),
        "l3": (a_J1, a_H),
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
    envelope_cfg,        # config.envelope
    prev_params,         # dict of (E,) tensors
):
    num_envs = lidar_points.shape[0]
    device = lidar_points.device

    margin = float(getattr(envelope_cfg, "margin_distance", 0.5))
    shrink = float(getattr(envelope_cfg, "shrink_step", 0.1))
    grow = float(getattr(envelope_cfg, "grow_step", 0.02))

    # 下界：对应底层 condition ranges 的 min，防止包络收缩到训练分布外
    # 全部从 config 读取（config 中 x3_min=-0.6 是数值上界，x3_max=-0.9 是数值下界）
    mins = {
        "x1": envelope_cfg.x1_min, "x3": envelope_cfg.x3_min,
        "l1": envelope_cfg.front_rear_min, "r1": envelope_cfg.front_rear_min,
        "l2": envelope_cfg.mid_min, "r2": envelope_cfg.mid_min,
        "l3": envelope_cfg.front_rear_min, "r3": envelope_cfg.front_rear_min,
    }
    maxes = {
        "x1": envelope_cfg.x1_max, "x3": envelope_cfg.x3_max,
        "l1": envelope_cfg.front_rear_max, "r1": envelope_cfg.front_rear_max,
        "l2": envelope_cfg.mid_max, "r2": envelope_cfg.mid_max,
        "l3": envelope_cfg.front_rear_max, "r3": envelope_cfg.front_rear_max,
    }

    params = {k: prev_params[k].clone() for k in maxes}

    # Pre-create clamp bounds outside loop to avoid per-iteration tensor creation
    mins_t = {k: torch.tensor(v, device=device) for k, v in mins.items()}
    maxes_t = {k: torch.tensor(v, device=device) for k, v in maxes.items()}

    pts = lidar_points               # (E, N, 3) — body-frame, use XY for polygon, Z for 3D filter
    angles = torch.atan2(pts[..., 1], pts[..., 0])  # (E, N)  [-pi, pi]

    inner = _build_hex_edges(
        params["x1"], params["x3"],
        params["l1"], params["l2"], params["l3"],
        params["r1"], params["r2"], params["r3"],
    )
    outer = _offset_hexagon(inner, margin)

    sectors = _sector_angles(inner)

    outside_inner = ~_point_inside_convex(pts, inner)  # (E, N)  XY only
    inside_outer = _point_inside_convex(pts, outer)     # (E, N)  XY only
    in_zone_xy = outside_inner & inside_outer            # (E, N)  XY in band

    # Z-dimension filter: points must be within the 3D prism height
    z_bottom = float(envelope_cfg.z_bottom)
    z_top = float(envelope_cfg.z_top)
    pts_z = pts[..., 2]                                   # (E, N)
    in_zone_z = (pts_z >= z_bottom) & (pts_z <= z_top)    # (E, N)

    in_zone = in_zone_xy & in_zone_z                      # (E, N)  3D volume

    for name, (s_start, s_end) in sectors.items():
        in_sector = _angle_in_range(angles, s_start.unsqueeze(-1), s_end.unsqueeze(-1))
        hit = (in_zone & in_sector).any(dim=-1)

        if name == "x3":
            # x3 为负值：hit → +shrink 向 x3_min(-0.6, 最近后方) 收缩;
            # no-hit → -grow 向 x3_max(-0.9, 最远后方) 恢复
            # clamp 下界=maxes["x3"](-0.9), 上界=mins["x3"](-0.6)
            delta = torch.where(hit, shrink, -grow)
            new_val = torch.clamp(params[name] + delta,
                                 maxes_t[name],
                                 mins_t[name])
        else:
            delta = torch.where(hit, -shrink, grow)
            new_val = torch.clamp(params[name] + delta,
                                 mins_t[name],
                                 maxes_t[name])
        params[name] = new_val

    return params


def envelope_params_to_condition(params, cfg):
    """Convert 8 independent envelope parameters to 8-dim condition tensor.

    Aggregates left/right widths via min(), clips to training ranges,
    and computes morphology priors via directional_ratio formula.

    Args:
        params: dict of (E,) tensors — x1, x3, l1, r1, l2, r2, l3, r3
        cfg: full env config (El4090EACfg instance)

    Returns:
        (E, 8) tensor ordered as cfg.commands.condition_names
    """
    # -- 1. 聚合左右为 5 长度 --
    front_width = torch.min(params["l1"], params["r1"])
    middle_width = torch.min(params["l2"], params["r2"])
    back_width = torch.min(params["l3"], params["r3"])
    forward_limit = params["x1"]
    backward_limit = params["x3"]

    # -- 2. Clip 到 command_ranges --
    ranges = cfg.commands.ranges
    front_width = _clip_range(front_width, *ranges.front_width)
    middle_width = _clip_range(middle_width, *ranges.middle_width)
    back_width = _clip_range(back_width, *ranges.back_width)
    forward_limit = _clip_range(forward_limit, *ranges.forward_limit)
    backward_limit = _clip_range(backward_limit, *ranges.backward_limit)

    # -- 3. 归一化 [0, 1] --
    fw_norm = _norm(front_width, *ranges.front_width)
    mw_norm = _norm(middle_width, *ranges.middle_width)
    bw_norm = _norm(back_width, *ranges.back_width)
    fwd_norm = _norm(forward_limit, *ranges.forward_limit)
    # backward_limit: flip sign first, then normalize
    bwd_norm = _norm(-backward_limit,
                     -ranges.backward_limit[1],  # -(-0.6) = 0.6
                     -ranges.backward_limit[0])  # -(-0.9) = 0.9

    # -- 4. directional_ratio 先验 --
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

    # -- 5. Clamp priors --
    front_prior = torch.clamp(front_prior, 0.0, 1.0)
    middle_prior = torch.clamp(middle_prior, 0.0, 1.0)
    back_prior = torch.clamp(back_prior, 0.0, 1.0)

    # -- 6. 按 condition_names 顺序输出 --
    return torch.stack([
        front_width, middle_width, back_width,
        forward_limit, backward_limit,
        front_prior, middle_prior, back_prior,
    ], dim=-1)  # (E, 8)


def _clip_range(tensor, low, high):
    return torch.clamp(tensor, min=low, max=high)


def _norm(tensor, low, high):
    return torch.clamp((tensor - low) / max(high - low, 1e-6), 0.0, 1.0)
