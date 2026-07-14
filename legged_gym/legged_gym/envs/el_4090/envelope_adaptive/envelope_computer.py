"""Fan-sector point cloud detection + step-based envelope shrink/grow."""

import torch

_debug_step = 0
_DEBUG_INTERVAL = 20  # print every N calls


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

    maxes = {
        "x1": envelope_cfg.x1_max, "x3": envelope_cfg.x3_max,
        "l1": envelope_cfg.front_rear_max, "r1": envelope_cfg.front_rear_max,
        "l2": envelope_cfg.mid_max, "r2": envelope_cfg.mid_max,
        "l3": envelope_cfg.front_rear_max, "r3": envelope_cfg.front_rear_max,
    }

    params = {k: prev_params[k].clone() for k in maxes}

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

    global _debug_step
    _debug_step += 1
    # if _debug_step % _DEBUG_INTERVAL == 0:
    #     n_zone = in_zone.sum().item()
    #     outer_x = outer[0, :, 0]; outer_y = outer[0, :, 1]
    #     print(f"[step {_debug_step}] outer x:[{outer_x.min():.3f},{outer_x.max():.3f}] "
    #           f"y:[{outer_y.min():.3f},{outer_y.max():.3f}]  zone pts: {n_zone}")
    #     if n_zone > 0:
    #         zone_pts = pts[0][in_zone[0]]
    #         dists_zone = torch.norm(zone_pts[:, :2], dim=-1)
    #         far_idx = dists_zone.argmax()
    #         fp = zone_pts[far_idx]
    #         fp_xy = fp[:2].reshape(1, 1, 2)
    #         print(f"  farthest zone pt: xy=({fp[0]:.2f},{fp[1]:.2f}) z={fp[2]:.2f} dist={dists_zone[far_idx]:.2f}")
    #         print(f"  inside_outer? {_point_inside_convex(fp_xy, outer)[0,0].item()}  "
    #               f"inside_inner? {_point_inside_convex(fp_xy, inner)[0,0].item()}  "
    #               f"in_Z? {bool((fp[2] >= z_bottom and fp[2] <= z_top).item())}")
    #         zone_dists = dists_zone
    #         print(f"  zone dist min/mean/max: {zone_dists.min():.2f}/{zone_dists.mean():.2f}/{zone_dists.max():.2f}  "
    #               f"Z min/max: {zone_pts[:,2].min():.2f}/{zone_pts[:,2].max():.2f}  "
    #               f"x1={params['x1'][0].item():.3f} r2={params['r2'][0].item():.3f}")

    _hit_names = []
    for name, (s_start, s_end) in sectors.items():
        in_sector = _angle_in_range(angles, s_start.unsqueeze(-1), s_end.unsqueeze(-1))
        hit = (in_zone & in_sector).any(dim=-1)
        if hit[0].item():
            _hit_names.append(name)

        if name == "x3":
            # x3 is negative -- shrink = increase (toward 0), grow = decrease
            delta = torch.where(hit, shrink, -grow)
            new_val = torch.clamp(params[name] + delta,
                                 torch.tensor(maxes[name], device=device),
                                 torch.zeros_like(params[name]))
        else:
            delta = torch.where(hit, -shrink, grow)
            new_val = torch.clamp(params[name] + delta,
                                 torch.zeros_like(params[name]),
                                 torch.tensor(maxes[name], device=device))
        params[name] = new_val

    # if _debug_step % _DEBUG_INTERVAL == 0:
    #     print(f"          hits: {_hit_names if _hit_names else 'none'}")

    return params
