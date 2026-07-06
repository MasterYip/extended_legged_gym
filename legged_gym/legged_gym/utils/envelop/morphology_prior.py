import argparse
import ast
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from draw_envelope import (
    draw_mammal_robot,
    load_el4090_config,
    load_robot_triangles,
    parse_urdf,
    resolve_asset_path,
    set_axes_equal,
    transform_matrix,
)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
CONFIG_PATH = PROJECT_ROOT / "legged_gym" / "envs" / "el_4090" / "spider_envelop" / "el4090_spider_config.py"

DEFAULT_WEIGHTS = {
    "front": {"front_width": 0.5, "forward_limit": 0.5},
    "middle": {"middle_width": 1.0},
    "back": {"back_width": 0.5, "backward_limit": 0.5},
}


def find_class(node, name):
    for child in node.body:
        if isinstance(child, ast.ClassDef) and child.name == name:
            return child
    raise KeyError(f"Class {name} not found")


def literal_assigns(class_node):
    values = {}
    for item in class_node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            try:
                values[item.targets[0].id] = ast.literal_eval(item.value)
            except (ValueError, SyntaxError):
                pass
    return values


def load_command_config(config_path=CONFIG_PATH):
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    cfg_class = find_class(tree, "El4090EnvelopCfg")
    commands_class = find_class(cfg_class, "commands")
    ranges_class = find_class(commands_class, "ranges")

    commands = literal_assigns(commands_class)
    ranges = literal_assigns(ranges_class)
    condition_names = commands.get(
        "condition_names",
        [
            "front_width",
            "middle_width",
            "back_width",
            "forward_limit",
            "backward_limit",
            "morphology_front_prior",
            "morphology_middle_prior",
            "morphology_back_prior",
        ],
    )
    weights = commands.get("morphology_prior_weights", DEFAULT_WEIGHTS)

    for name in ("morphology_prior", "morphology_front_prior", "morphology_middle_prior", "morphology_back_prior"):
        if name in condition_names and name not in ranges:
            ranges[name] = [0.0, 1.0]

    missing = [name for name in condition_names if name not in ranges]
    if missing:
        raise ValueError(f"Missing condition ranges in {config_path}: {missing}")
    return condition_names, ranges, weights


def condition_bounds(condition_names, ranges):
    low = np.array([ranges[name][0] for name in condition_names], dtype=np.float64)
    high = np.array([ranges[name][1] for name in condition_names], dtype=np.float64)
    return low, high


def normalize_condition_value(condition, condition_names, low, high, name):
    idx = condition_names.index(name)
    value = condition[:, idx]
    value_low = low[idx]
    value_high = high[idx]
    if name == "backward_limit":
        value = -value
        value_low = -high[idx]
        value_high = -low[idx]
    return np.clip((value - value_low) / max(value_high - value_low, 1e-6), 0.0, 1.0)


def set_morphology_prior_from_envelope(condition, condition_names, low, high, weights):
    prior_names = (
        "morphology_front_prior",
        "morphology_middle_prior",
        "morphology_back_prior",
    )
    if not all(name in condition_names for name in prior_names):
        return condition

    prior_specs = {
        "morphology_front_prior": weights.get("front", {"front_width": 0.5, "forward_limit": 0.5}),
        "morphology_middle_prior": weights.get("middle", {"middle_width": 1.0}),
        "morphology_back_prior": weights.get("back", {"back_width": 0.5, "backward_limit": 0.5}),
    }
    for prior_name, prior_weights in prior_specs.items():
        prior = np.zeros(condition.shape[0], dtype=np.float64)
        weight_sum = 0.0
        for name, weight in prior_weights.items():
            weight = float(weight)
            prior += weight * normalize_condition_value(condition, condition_names, low, high, name)
            weight_sum += weight
        prior = np.clip(prior / max(weight_sum, 1e-6), 0.0, 1.0)
        condition[:, condition_names.index(prior_name)] = prior
    return condition


def sample_condition(condition_names, ranges, weights, samples, seed):
    if seed is not None:
        np.random.seed(seed)

    low, high = condition_bounds(condition_names, ranges)
    condition = low + np.random.rand(samples, len(condition_names)) * (high - low)
    condition = set_morphology_prior_from_envelope(condition, condition_names, low, high, weights)
    return condition


def print_condition_table(condition_names, condition):
    mins = condition.min(axis=0)
    maxs = condition.max(axis=0)
    means = condition.mean(axis=0)

    print("Sampled condition stats:")
    for idx, name in enumerate(condition_names):
        print(f"  {name}: min={mins[idx]:.6f}, mean={means[idx]:.6f}, max={maxs[idx]:.6f}")

    print("\nFirst sample:")
    for idx, name in enumerate(condition_names):
        print(f"  {name}: {condition[0, idx]:.6f}")


def condition_dict(condition_names, condition_row):
    values = {name: condition_row[idx] for idx, name in enumerate(condition_names)}
    if "front_width" in values:
        values["left_front"] = values["front_width"]
        values["right_front"] = -values["front_width"]
        values["left_mid"] = values["middle_width"]
        values["right_mid"] = -values["middle_width"]
        values["left_back"] = values["back_width"]
        values["right_back"] = -values["back_width"]
    return values


def envelope_points(values):
    return np.array(
        [
            [values["forward_limit"], values["left_front"]],
            [0.0, values["left_mid"]],
            [values["backward_limit"], values["left_back"]],
            [values["backward_limit"], values["right_back"]],
            [0.0, values["right_mid"]],
            [values["forward_limit"], values["right_front"]],
        ],
        dtype=np.float64,
    )


def build_prior_joint_angles(default_angles, mammal_angles, values):
    priors_by_leg = {
        "LF": values["morphology_front_prior"],
        "RF": values["morphology_front_prior"],
        "LM": values["morphology_middle_prior"],
        "RM": values["morphology_middle_prior"],
        "LB": values["morphology_back_prior"],
        "RB": values["morphology_back_prior"],
    }
    joint_angles = {}
    for name, spider_angle in default_angles.items():
        leg_name = name.split("_", 1)[0]
        prior = priors_by_leg.get(leg_name, 0.0)
        mammal_angle = mammal_angles[name]
        joint_angles[name] = spider_angle + prior * (mammal_angle - spider_angle)
    return joint_angles


def envelope_prism(values, z_bottom=-0.42, z_top=0.02):
    footprint = envelope_points(values)
    bottom = np.column_stack((footprint, np.full(len(footprint), z_bottom)))
    top = np.column_stack((footprint, np.full(len(footprint), z_top)))
    return footprint, bottom, top


def draw_envelope_prism(ax, values, z_bottom=-0.42, z_top=0.02):
    footprint, bottom, top = envelope_prism(values, z_bottom=z_bottom, z_top=z_top)
    side_faces = []
    for i in range(len(footprint)):
        j = (i + 1) % len(footprint)
        side_faces.append([bottom[i], bottom[j], top[j], top[i]])

    collection = Poly3DCollection(
        side_faces + [top],
        alpha=0.16,
        facecolor="#6EC6FF",
        edgecolor="#0077B6",
        linewidth=1.4,
    )
    ax.add_collection3d(collection)

    for ring, color in ((bottom, "#0077B6"), (top, "#D1495B")):
        closed = np.vstack((ring, ring[0]))
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linewidth=2.0)
    for i in range(len(footprint)):
        ax.plot(
            [bottom[i, 0], top[i, 0]],
            [bottom[i, 1], top[i, 1]],
            [bottom[i, 2], top[i, 2]],
            color="#0077B6",
            linewidth=1.0,
            alpha=0.75,
        )

    leg_specs = [
        ("front", values["forward_limit"], values["left_front"], values["morphology_front_prior"], "#D1495B"),
        ("middle", 0.0, values["left_mid"], values["morphology_middle_prior"], "#2A9D8F"),
        ("back", values["backward_limit"], values["left_back"], values["morphology_back_prior"], "#F4A261"),
    ]
    for label, x, y_abs, prior, color in leg_specs:
        ax.plot([x, x], [-y_abs, y_abs], [z_top, z_top], color=color, linewidth=3.0)
        ax.scatter([x, x], [y_abs, -y_abs], [z_top, z_top], s=45, color=color, edgecolor="black")
        ax.text(x, y_abs, z_top + 0.035, f"{label}={prior:.2f}", color=color, ha="center")

    return np.vstack((bottom, top))


def plot_condition_3d(condition_names, condition_row, output, show=True, max_triangles_per_link=700, seed=4090):
    values = condition_dict(condition_names, condition_row)
    asset_cfg, init_state = load_el4090_config()
    default_angles = init_state["default_joint_angles"]
    mammal_angles = init_state["mammal_default_joint_angles"]
    joint_angles = build_prior_joint_angles(default_angles, mammal_angles, values)

    urdf_path = resolve_asset_path(asset_cfg["file"])
    rng = np.random.RandomState(seed)
    links, joints = parse_urdf(urdf_path)
    robot_triangles = load_robot_triangles(links, max_triangles_per_link, rng)
    root_tf = transform_matrix(np.array(init_state["pos"], dtype=float), np.zeros(3))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("EL4090 Sampled 3D Envelope And Morphology-Prior Robot")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=22, azim=-135)

    envelope_points_3d = draw_envelope_prism(ax, values)
    draw_mammal_robot(ax, links, joints, robot_triangles, joint_angles, root_tf)

    robot_axis_points = np.array(
        [
            [-0.95, -0.8, -0.55],
            [0.95, 0.8, 0.25],
        ],
        dtype=float,
    )
    set_axes_equal(ax, np.vstack((envelope_points_3d, robot_axis_points)))
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=220)
        print(f"\nSaved 3D visualization: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_condition(condition_names, condition_row, output, show=False):
    values = condition_dict(condition_names, condition_row)
    points = envelope_points(values)
    closed = np.vstack((points, points[0]))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.fill(points[:, 0], points[:, 1], color="#6EC6FF", alpha=0.18)
    ax.plot(closed[:, 0], closed[:, 1], color="#0077B6", linewidth=2.0, label="sampled envelope")
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="#555555", linewidth=0.8, alpha=0.5)

    leg_specs = [
        ("front", values["forward_limit"], values["left_front"], values["morphology_front_prior"], "#D1495B"),
        ("middle", 0.0, values["left_mid"], values["morphology_middle_prior"], "#2A9D8F"),
        ("back", values["backward_limit"], values["left_back"], values["morphology_back_prior"], "#F4A261"),
    ]
    for label, x, y_abs, prior, color in leg_specs:
        ax.scatter([x, x], [y_abs, -y_abs], s=90, color=color, edgecolor="black", zorder=4)
        ax.plot([x, x], [-y_abs, y_abs], color=color, linewidth=2.5, alpha=0.7)
        ax.text(
            x,
            y_abs + 0.045,
            f"{label} prior={prior:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )

    ax.set_title("Sampled Envelope And Segment Morphology Priors")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    margin = 0.15
    ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=220)
        print(f"\nSaved visualization: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Sample EL4090 envelope conditions and infer morphology_prior with the same logic used by spider_envelop."
    )
    parser.add_argument("--samples", type=int, default=8, help="Number of condition samples to generate.")
    parser.add_argument("--seed", type=int, default=4090, help="Random seed. Use -1 for non-deterministic sampling.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to el4090_spider_config.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "morphology_prior_3d.png",
        help="Path to save the visualization for the first sample. Use 'none' to disable.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show the visualization window.")
    parser.add_argument("--plot-2d", action="store_true", help="Draw the older 2D footprint view instead of the 3D view.")
    parser.add_argument(
        "--max-triangles-per-link",
        type=int,
        default=700,
        help="Downsample rendered robot triangles for display speed.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    condition_names, ranges, weights = load_command_config(args.config)
    seed = None if args.seed < 0 else args.seed
    condition = sample_condition(condition_names, ranges, weights, args.samples, seed)
    print_condition_table(condition_names, condition)
    output = None if str(args.output).lower() == "none" else args.output
    show = not args.no_show
    if args.plot_2d:
        plot_condition(condition_names, condition[0], output, show=show)
    else:
        plot_condition_3d(
            condition_names,
            condition[0],
            output,
            show=show,
            max_triangles_per_link=args.max_triangles_per_link,
            seed=4090 if seed is None else seed,
        )


if __name__ == "__main__":
    main()
