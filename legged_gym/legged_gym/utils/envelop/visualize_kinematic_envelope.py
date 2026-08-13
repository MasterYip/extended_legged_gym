"""Generate the ENV-DESIGN-003 academic triptych from the tested Torch API."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
import torch

from kinematic_envelope import (
    BatchedUrdfKinematics,
    add_support_margin,
    capsule_support,
    default_el4090_capsules,
    foot_positions,
    legacy_condition_from_support,
    load_urdf_joints,
    support_directions,
)


PAPER = "#F7F9FA"
GRAPHITE = "#20262E"
TEAL = "#007C83"
CYAN = "#38BFC3"
RED = "#D95043"
AMBER = "#D99A2B"


def halfspace_polygon(directions: np.ndarray, support: np.ndarray) -> np.ndarray:
    polygon = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])
    for normal, bound in zip(directions, support):
        clipped = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            start_in = normal @ start <= bound + 1e-10
            end_in = normal @ end <= bound + 1e-10
            if start_in:
                clipped.append(start)
            if start_in != end_in:
                delta = end - start
                clipped.append(start + delta * ((bound - normal @ start) / (normal @ delta)))
        polygon = np.asarray(clipped)
    return polygon


def datum(ax) -> None:
    ax.axhline(0, color=GRAPHITE, lw=0.65, alpha=0.38)
    ax.axvline(0, color=GRAPHITE, lw=0.65, alpha=0.38)
    ax.annotate("+x forward", (0.95, 0), xytext=(-5, 7), textcoords="offset points", ha="right", fontsize=8)
    ax.annotate("+y left", (0, 0.78), xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.88, 0.88)
    ax.set_xlabel("body-yaw x / m")
    ax.set_ylabel("body-yaw y / m")


def flow_box(ax, x, y, width, height, title, detail, color):
    patch = plt.Rectangle((x, y), width, height, facecolor=PAPER, edgecolor=color, lw=1.5)
    ax.add_patch(patch)
    title_artist = ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", color=GRAPHITE, weight="semibold", fontsize=6.5)
    detail_artist = ax.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center", color=GRAPHITE, fontsize=6.5, family="DejaVu Sans Mono")
    return patch, (title_artist, detail_artist)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("utils/envelop/figures"))
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    urdf = root / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
    kin = BatchedUrdfKinematics(load_urdf_joints(urdf))
    q = torch.tensor([[0.0, 0.6, -0.6] * 6], dtype=torch.float64)
    directions = support_directions(args.directions, dtype=q.dtype)
    support = capsule_support(kin, q, default_el4090_capsules(), directions)[0]
    safe = add_support_margin(support, args.margin)
    feet = foot_positions(kin, q)[0].detach().numpy()
    polygon = halfspace_polygon(directions.numpy(), support.detach().numpy())
    safe_polygon = halfspace_polygon(directions.numpy(), safe.detach().numpy())
    condition = legacy_condition_from_support(support.unsqueeze(0), directions)[0]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": GRAPHITE,
        "axes.labelcolor": GRAPHITE,
        "text.color": GRAPHITE,
        "xtick.color": GRAPHITE,
        "ytick.color": GRAPHITE,
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
    })
    fig = plt.figure(figsize=(14.2, 4.65), facecolor=PAPER)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.05, 1.3), wspace=0.27)

    ax = fig.add_subplot(grid[0, 0])
    datum(ax)
    ax.add_patch(Polygon(safe_polygon, closed=True, facecolor=CYAN, edgecolor=TEAL, alpha=0.16, lw=1.1, label="safe envelope"))
    ax.add_patch(Polygon(polygon, closed=True, facecolor=TEAL, edgecolor=TEAL, alpha=0.24, lw=1.6, label="occupied capsules"))
    ax.scatter(feet[:, 0], feet[:, 1], c=AMBER, s=24, zorder=4, label="feet at reference pose")
    ax.add_patch(plt.Rectangle((-0.45, -0.12), 0.9, 0.24, facecolor=GRAPHITE, alpha=0.72))
    ax.set_title("A  Body-yaw geometry", loc="left")
    ax.legend(loc="lower left", fontsize=7, frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    datum(ax)
    ax.add_patch(Polygon(safe_polygon, closed=True, fill=False, edgecolor=CYAN, lw=2.0))
    ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=TEAL, lw=1.4))
    selected = [2, 8, 17]
    for idx in selected:
        normal = directions[idx].numpy()
        boundary = normal * support[idx].item()
        safe_boundary = normal * safe[idx].item()
        ax.add_patch(FancyArrowPatch((0, 0), boundary, arrowstyle="-|>", mutation_scale=9, color=TEAL, lw=1.2))
        ax.plot([boundary[0], safe_boundary[0]], [boundary[1], safe_boundary[1]], color=RED, lw=2.2)
    ax.text(-0.96, -0.79, r"$h_k^{safe}=h_k^{occ}+m$", fontsize=12, color=RED)
    ax.text(0.12, 0.54, f"m = {args.margin:.2f} m", fontsize=8, color=RED)
    ax.set_title("B  Fixed-direction support", loc="left")

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    labeled_boxes = [
        flow_box(ax, 0.2, 4.8, 2.1, 1.2, "Body LiDAR", "[B, N, 3]", RED),
        flow_box(ax, 3.0, 4.8, 2.1, 1.2, "Point query", "Up <= h + m", TEAL),
        flow_box(ax, 5.8, 4.8, 2.1, 1.2, "Geometry state", "support [B, K]", CYAN),
        flow_box(ax, 3.0, 2.3, 2.1, 1.2, "Joint export", "center/range [B,6]", AMBER),
        flow_box(ax, 5.8, 2.3, 2.1, 1.2, "Legacy adapter", "condition [B,8]", GRAPHITE),
        flow_box(ax, 7.8, 0.25, 2.0, 1.2, "RL observation", "unchanged [B,83]", TEAL),
    ]
    arrows = [((2.3, 5.4), (3.0, 5.4)), ((5.1, 5.4), (5.8, 5.4)), ((6.85, 4.8), (4.05, 3.5)), ((5.1, 2.9), (5.8, 2.9)), ((7.9, 2.9), (8.8, 1.45))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=GRAPHITE, lw=1.0))
    ax.text(0.2, 6.62, "C  Perception-to-policy contract", fontsize=11, weight="semibold")
    ax.text(0.25, 0.35, "Projection remains approximate\nCapsules are not mesh-complete", fontsize=8, color=RED)
    ax.text(5.84, 4.47, f"legacy: [{condition[0]:.2f}, {condition[1]:.2f}, ...]", fontsize=7, family="DejaVu Sans Mono")

    fig.suptitle("EL4090 kinematic envelope: auditable geometry and unchanged policy interface", x=0.03, ha="left", fontsize=15, family="STIXGeneral")
    fig.text(0.03, 0.015, "Occupied capsule support and reachable-foot geometry are modeled separately. +x forward, +y left.", fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    for artist in fig.findobj(match=lambda item: isinstance(item, plt.Text) and item.get_visible()):
        bounds = artist.get_window_extent(renderer)
        if bounds.width and bounds.height and (
            not figure_box.contains(*bounds.get_points()[0])
            or not figure_box.contains(*bounds.get_points()[1])
        ):
            raise RuntimeError(f"Text lies outside figure canvas: {artist.get_text()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("png", {"dpi": 220}), ("svg", {}), ("pdf", {})):
        fig.savefig(args.output_dir / f"kinematic_envelope_triptych.{extension}", bbox_inches="tight", facecolor=PAPER, **kwargs)
    svg_path = args.output_dir / "kinematic_envelope_triptych.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
