"""Train/load and visualize the EL4090 HAA swing-range network.

Run from ``legged_gym/legged_gym``:

    python utils/envelop/network/visualize_haa_swing_range.py \
        --output utils/envelop/network/haa_swing_range_visualization.png

When ``--checkpoint`` is omitted, a network is trained in memory from the
analytical estimator.  Use ``--save-checkpoint`` to keep that trained model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/legged_gym_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor

ENVELOP_DIR = Path(__file__).resolve().parent.parent
if str(ENVELOP_DIR) not in sys.path:
    sys.path.insert(0, str(ENVELOP_DIR))

from draw_envelope import (
    compute_link_transforms,
    draw_mammal_robot,
    load_robot_triangles,
    parse_urdf,
    resolve_asset_path,
    set_axes_equal,
    transform_matrix,
)
from morphology_prior import (
    build_prior_joint_angles,
    condition_dict,
    draw_envelope_prism,
    load_el4090_config,
)

if TYPE_CHECKING:
    from .haa_swing_range import (
        AnalyticHaaRangeEstimator,
        HaaRangeConfig,
        HaaRangeNetwork,
        MonteCarloHaaRangeEstimator,
        apply_env_morphology_priors,
        load_envelope_condition_spec,
    )
else:
    try:
        from .haa_swing_range import (
            AnalyticHaaRangeEstimator,
            HaaRangeConfig,
            HaaRangeNetwork,
            MonteCarloHaaRangeEstimator,
            apply_env_morphology_priors,
            load_envelope_condition_spec,
        )
    except ImportError:
        from haa_swing_range import (
            AnalyticHaaRangeEstimator,
            HaaRangeConfig,
            HaaRangeNetwork,
            MonteCarloHaaRangeEstimator,
            apply_env_morphology_priors,
            load_envelope_condition_spec,
        )


DEMO_CONDITION = (0.45, 0.50, 0.45, 0.75, -0.75, 0.5, 0.5, 0.5)


def obtain_network(
    device: torch.device,
    *,
    checkpoint: Optional[Path],
    train_steps: int,
    batch_size: int,
    label_method: str,
    monte_carlo_samples: int,
) -> Tuple[HaaRangeNetwork, float]:
    """Load a network or train one and return its validation MAE."""
    condition_spec = load_envelope_condition_spec()
    config = HaaRangeConfig(condition_names=condition_spec.condition_names)
    if checkpoint is not None:
        network = HaaRangeNetwork.from_checkpoint(checkpoint, device=device)
    else:
        network = HaaRangeNetwork(config).to(device)
        teacher: AnalyticHaaRangeEstimator
        if label_method == "monte_carlo":
            teacher = MonteCarloHaaRangeEstimator(
                config,
                num_samples=monte_carlo_samples,
                seed=4090,
            )
        else:
            teacher = AnalyticHaaRangeEstimator(config)
        low, high = condition_spec.bounds(device)
        final_mse = network.fit_estimator(
            low,
            high,
            teacher,
            steps=train_steps,
            batch_size=batch_size,
            condition_transform=lambda value: apply_env_morphology_priors(value, condition_spec),
        )
        print(f"training_final_mse={final_mse:.8f}")

    low, high = condition_spec.bounds(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(4090)
    validation = low + torch.rand(
        2048,
        low.numel(),
        generator=generator,
        device=device,
    ) * (high - low)
    validation = apply_env_morphology_priors(validation, condition_spec)
    analytic = AnalyticHaaRangeEstimator(config)(validation)
    network.eval()
    with torch.no_grad():
        prediction = network(validation)
    validation_mae = float(torch.mean(torch.abs(prediction - analytic)).cpu())
    return network, validation_mae


def draw_interval_comparison(
    ax,
    config: HaaRangeConfig,
    analytic: Tensor,
    sampled: Tensor,
    predicted: Tensor,
) -> None:
    methods = (
        ("analytic", analytic, "#0077B6", 0.18),
        ("Monte Carlo", sampled, "#F4A261", 0.0),
        ("network", predicted, "#D62828", -0.18),
    )
    y_base = torch.arange(len(config.leg_names), dtype=torch.float32)
    for label, ranges, color, offset in methods:
        ranges = ranges.detach().cpu()
        y = y_base + offset
        ax.hlines(
            y.numpy(),
            ranges[:, 0].numpy(),
            ranges[:, 1].numpy(),
            color=color,
            linewidth=3.0,
            label=label,
        )
        ax.scatter(ranges[:, 0].numpy(), y.numpy(), color=color, s=22)
        ax.scatter(ranges[:, 1].numpy(), y.numpy(), color=color, s=22)
    ax.set_yticks(y_base.numpy(), config.leg_names)
    ax.set_xlabel("HAA angle [rad]")
    ax.set_title("One condition: lower/upper range for every leg")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=3, loc="upper center")


def draw_prior_sweep(
    ax,
    network: HaaRangeNetwork,
    config: HaaRangeConfig,
    group_name: str,
    leg_index: int,
    prior_index: int,
    device: torch.device,
    monte_carlo_samples: int,
) -> None:
    prior = torch.linspace(0.0, 1.0, 101, device=device)
    condition = torch.tensor(DEMO_CONDITION, device=device).repeat(prior.numel(), 1)
    condition[:, prior_index] = prior
    analytic_estimator = AnalyticHaaRangeEstimator(config)
    analytic = analytic_estimator(condition)[:, leg_index]
    with torch.no_grad():
        predicted = network(condition)[:, leg_index]

    prior_cpu = prior.cpu()
    analytic_cpu = analytic.cpu()
    predicted_cpu = predicted.cpu()
    ax.fill_between(
        prior_cpu.numpy(),
        analytic_cpu[:, 0].numpy(),
        analytic_cpu[:, 1].numpy(),
        color="#0077B6",
        alpha=0.16,
        label="analytic feasible region",
    )
    ax.plot(prior_cpu.numpy(), analytic_cpu[:, 0].numpy(), color="#0077B6", linewidth=1.7)
    ax.plot(prior_cpu.numpy(), analytic_cpu[:, 1].numpy(), color="#0077B6", linewidth=1.7)
    ax.plot(
        prior_cpu.numpy(),
        predicted_cpu[:, 0].numpy(),
        "--",
        color="#D62828",
        linewidth=1.5,
        label="network",
    )
    ax.plot(prior_cpu.numpy(), predicted_cpu[:, 1].numpy(), "--", color="#D62828", linewidth=1.5)

    sample_indices = torch.linspace(0, prior.numel() - 1, 11, device=device).long()
    sampled = MonteCarloHaaRangeEstimator(
        config,
        num_samples=monte_carlo_samples,
        seed=4090 + leg_index,
    )(condition[sample_indices])[:, leg_index].cpu()
    sample_prior = prior[sample_indices].cpu()
    ax.scatter(sample_prior.numpy(), sampled[:, 0].numpy(), color="#F4A261", marker="x", s=26)
    ax.scatter(
        sample_prior.numpy(),
        sampled[:, 1].numpy(),
        color="#F4A261",
        marker="x",
        s=26,
        label="Monte Carlo",
    )
    ax.set_title(f"{group_name} leg ({config.leg_names[leg_index]})")
    ax.set_xlabel("morphology prior")
    ax.set_ylabel("HAA bound [rad]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def create_diagnostic_2d(
    network: HaaRangeNetwork,
    output: Path,
    *,
    device: torch.device,
    validation_mae: float,
    monte_carlo_samples: int,
) -> None:
    config = network.config
    condition = torch.tensor(DEMO_CONDITION, dtype=torch.float32, device=device).unsqueeze(0)
    analytic = AnalyticHaaRangeEstimator(config)(condition)[0]
    sampled = MonteCarloHaaRangeEstimator(
        config,
        num_samples=monte_carlo_samples,
        seed=4090,
    )(condition)[0]
    with torch.no_grad():
        predicted = network(condition)[0]

    figure = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.05, 1.0))
    interval_ax = figure.add_subplot(grid[0, :])
    draw_interval_comparison(interval_ax, config, analytic, sampled, predicted)

    sweep_specs = (
        ("front", 0, 5),
        ("middle", 1, 6),
        ("back", 2, 7),
    )
    for column, (group_name, leg_index, prior_index) in enumerate(sweep_specs):
        draw_prior_sweep(
            figure.add_subplot(grid[1, column]),
            network,
            config,
            group_name,
            leg_index,
            prior_index,
            device,
            monte_carlo_samples,
        )

    condition_text = (
        f"demo widths F/M/B={DEMO_CONDITION[0]:.2f}/{DEMO_CONDITION[1]:.2f}/{DEMO_CONDITION[2]:.2f}, "
        f"reach front/back={DEMO_CONDITION[3]:.2f}/{DEMO_CONDITION[4]:.2f}, "
        f"priors F/M/B={DEMO_CONDITION[5]:.2f}/{DEMO_CONDITION[6]:.2f}/{DEMO_CONDITION[7]:.2f}"
    )
    figure.suptitle(
        "EL4090 envelope-conditioned HAA range network\n"
        f"validation MAE vs analytic = {validation_mae:.5f} rad | {condition_text}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    print(f"saved_visualization={output}")


def leg_chain_points(link_transforms, leg_name: str) -> np.ndarray:
    """Return hip, thigh, shank and foot frame origins for one leg."""
    link_names = (
        f"{leg_name}_HIP",
        f"{leg_name}_THIGH",
        f"{leg_name}_SHANK",
        f"{leg_name}_FOOT",
    )
    missing = [name for name in link_names if name not in link_transforms]
    if missing:
        raise KeyError(f"Missing link transforms for {leg_name}: {missing}")
    return np.stack([link_transforms[name][:3, 3] for name in link_names], axis=0)


def sample_leg_sweep(
    joints,
    joint_angles,
    root_transform: np.ndarray,
    leg_name: str,
    lower: float,
    upper: float,
    sample_count: int,
) -> np.ndarray:
    """Return ``[samples, 4, 3]`` kinematic chains across an HAA interval."""
    chains = []
    haa_name = f"{leg_name}_HAA"
    for angle in np.linspace(lower, upper, sample_count):
        sample_angles = dict(joint_angles)
        sample_angles[haa_name] = float(angle)
        transforms = compute_link_transforms(
            joints,
            sample_angles,
            root_transform=root_transform,
        )
        chains.append(leg_chain_points(transforms, leg_name))
    return np.stack(chains, axis=0)


def draw_leg_sweep(ax, chains: np.ndarray, color: str, leg_name: str, lower: float, upper: float) -> None:
    """Draw swept link surfaces, boundary poses and the resulting foot arc."""
    faces = []
    for sample_index in range(chains.shape[0] - 1):
        for segment_index in range(chains.shape[1] - 1):
            faces.append(
                [
                    chains[sample_index, segment_index],
                    chains[sample_index, segment_index + 1],
                    chains[sample_index + 1, segment_index + 1],
                    chains[sample_index + 1, segment_index],
                ]
            )
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor=color,
            alpha=0.13,
            linewidth=0.12,
        )
    )
    for boundary_chain in (chains[0], chains[-1]):
        ax.plot(
            boundary_chain[:, 0],
            boundary_chain[:, 1],
            boundary_chain[:, 2],
            color=color,
            linewidth=1.7,
            alpha=0.85,
        )
    foot_path = chains[:, -1]
    ax.plot(
        foot_path[:, 0],
        foot_path[:, 1],
        foot_path[:, 2],
        color=color,
        linewidth=3.0,
    )
    label_point = foot_path[len(foot_path) // 2]
    ax.text(
        label_point[0],
        label_point[1],
        label_point[2] + 0.035,
        f"{leg_name}\n[{lower:.2f}, {upper:.2f}]",
        color=color,
        fontsize=7,
        ha="center",
    )


class InteractiveHaaScene:
    """Cached robot geometry and estimators used by both static and UI rendering."""

    COLORS = ("#D62828", "#2A9D8F", "#F4A261", "#8E44AD", "#1D4ED8", "#7A5C00")
    METHOD_LABELS = {
        "analytic": "Analytic",
        "monte_carlo": "Monte Carlo",
        "network": "Network",
    }

    def __init__(
        self,
        network: HaaRangeNetwork,
        *,
        device: torch.device,
        monte_carlo_samples: int,
        max_triangles_per_link: int,
        seed: int,
    ) -> None:
        self.network = network
        self.device = device
        self.config = network.config
        self.condition_spec = load_envelope_condition_spec()
        if self.condition_spec.condition_names != self.config.condition_names:
            raise ValueError(
                "Network and spider_envelop condition orders differ: "
                f"{self.config.condition_names} != {self.condition_spec.condition_names}"
            )
        self.analytic = AnalyticHaaRangeEstimator(self.config)
        self.monte_carlo = MonteCarloHaaRangeEstimator(
            self.config,
            num_samples=monte_carlo_samples,
            seed=seed,
        )
        self.rng = np.random.RandomState(seed)

        self.asset_cfg, self.init_state = load_el4090_config()
        self.urdf_path = resolve_asset_path(self.asset_cfg["file"])
        self.links, self.joints = parse_urdf(self.urdf_path)
        self.robot_triangles = load_robot_triangles(
            self.links,
            max_triangles_per_link,
            self.rng,
        )
        self.root_transform = transform_matrix(
            np.asarray(self.init_state["pos"], dtype=float),
            np.zeros(3),
        )

    def condition_tensor(self, condition: Sequence[float]) -> Tensor:
        raw = torch.tensor(condition, dtype=torch.float32, device=self.device).unsqueeze(0)
        return apply_env_morphology_priors(raw, self.condition_spec)

    def default_condition(self) -> Tensor:
        return self.condition_tensor(DEMO_CONDITION)

    def random_condition(self) -> Tensor:
        low, high = self.condition_spec.bounds(self.device)
        random_values = torch.tensor(
            self.rng.rand(len(self.condition_spec.condition_names)),
            dtype=torch.float32,
            device=self.device,
        )
        condition = low + random_values * (high - low)
        return apply_env_morphology_priors(condition.unsqueeze(0), self.condition_spec)

    def ranges(self, condition: Tensor, method: str) -> Tensor:
        if method == "analytic":
            return self.analytic(condition)
        if method == "monte_carlo":
            return self.monte_carlo(condition)
        if method == "network":
            self.network.eval()
            with torch.no_grad():
                return self.network(condition)
        raise ValueError(f"Unknown HAA range method: {method!r}")

    def condition_summary(self, condition: Tensor) -> str:
        values = condition[0].detach().cpu().tolist()
        rows = [
            f"{name}: {value: .3f}"
            for name, value in zip(self.config.condition_names, values)
        ]
        return "\n".join(rows)

    def draw(
        self,
        ax,
        condition: Tensor,
        method: str,
        *,
        sweep_samples: int,
        validation_mae: float,
    ) -> None:
        """Clear and redraw the complete 3D scene in the same axes."""
        ax.cla()
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.view_init(elev=24, azim=-132)

        condition_cpu = condition[0].detach().cpu().numpy()
        values = condition_dict(self.config.condition_names, condition_cpu)
        joint_angles = build_prior_joint_angles(
            self.init_state["default_joint_angles"],
            self.init_state["mammal_default_joint_angles"],
            values,
        )
        predicted = self.ranges(condition, method)[0].detach().cpu().numpy()

        envelope_points_3d = draw_envelope_prism(ax, values)
        draw_mammal_robot(
            ax,
            self.links,
            self.joints,
            self.robot_triangles,
            joint_angles,
            self.root_transform,
        )

        sweep_points = []
        legend_handles = [
            Patch(facecolor="#6EC6FF", edgecolor="#0077B6", alpha=0.25, label="envelope"),
            Patch(facecolor="#777777", edgecolor="#333333", alpha=0.7, label="prior pose robot"),
        ]
        for leg_index, (leg_name, color) in enumerate(zip(self.config.leg_names, self.COLORS)):
            lower, upper = predicted[leg_index]
            chains = sample_leg_sweep(
                self.joints,
                joint_angles,
                self.root_transform,
                leg_name,
                float(lower),
                float(upper),
                sweep_samples,
            )
            draw_leg_sweep(ax, chains, color, leg_name, float(lower), float(upper))
            sweep_points.append(chains.reshape(-1, 3))
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=3, label=f"{leg_name} foot arc")
            )

        axis_support = np.array(((-0.95, -0.85, -0.55), (0.95, 0.85, 0.75)), dtype=float)
        set_axes_equal(ax, np.vstack((envelope_points_3d, *sweep_points, axis_support)))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.85, 0.85)
        ax.set_zlim(-0.58, 0.78)
        ax.set_box_aspect((2.0, 1.7, 1.35))
        method_label = self.METHOD_LABELS[method]
        mae_text = f"; network validation MAE={validation_mae:.5f} rad" if method == "network" else ""
        ax.set_title(
            f"EL4090 envelope and HAA swing ranges — {method_label}{mae_text}\n"
            "drag: rotate | mouse wheel: zoom | colored surfaces: leg sweeps",
            fontsize=11,
        )
        ax.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=2)


def _zoom_3d_axes(ax, scale: float) -> None:
    for getter, setter in (
        (ax.get_xlim3d, ax.set_xlim3d),
        (ax.get_ylim3d, ax.set_ylim3d),
        (ax.get_zlim3d, ax.set_zlim3d),
    ):
        lower, upper = getter()
        center = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower) * scale
        setter(center - half_width, center + half_width)


def launch_interactive_ui(
    network: HaaRangeNetwork,
    *,
    device: torch.device,
    validation_mae: float,
    monte_carlo_samples: int,
    sweep_samples: int,
    max_triangles_per_link: int,
    seed: int,
    initial_method: str,
    show: bool = True,
):
    """Create a Matplotlib UI and optionally block in ``plt.show()``."""
    scene = InteractiveHaaScene(
        network,
        device=device,
        monte_carlo_samples=monte_carlo_samples,
        max_triangles_per_link=max_triangles_per_link,
        seed=seed,
    )
    figure = plt.figure(figsize=(15, 9))
    figure.subplots_adjust(left=0.22, right=0.98, bottom=0.08, top=0.92)
    ax = figure.add_subplot(111, projection="3d")
    condition = scene.default_condition()
    state = {"condition": condition, "method": initial_method}

    random_button_ax = figure.add_axes((0.025, 0.83, 0.15, 0.055))
    random_button = Button(random_button_ax, "Random Envelope", color="#DCEEFF", hovercolor="#B9DCFF")
    method_ax = figure.add_axes((0.025, 0.57, 0.15, 0.20))
    method_radio = RadioButtons(
        method_ax,
        ("Analytic", "Monte Carlo", "Network"),
        active=("analytic", "monte_carlo", "network").index(initial_method),
    )
    save_button_ax = figure.add_axes((0.025, 0.49, 0.15, 0.055))
    save_button = Button(save_button_ax, "Save Snapshot", color="#E8F5E9", hovercolor="#C8E6C9")
    condition_text = figure.text(
        0.025,
        0.445,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )
    status_text = figure.text(0.025, 0.035, "Ready", fontsize=8, color="#444444")

    label_to_method = {
        "Analytic": "analytic",
        "Monte Carlo": "monte_carlo",
        "Network": "network",
    }

    def redraw() -> None:
        scene.draw(
            ax,
            state["condition"],
            state["method"],
            sweep_samples=sweep_samples,
            validation_mae=validation_mae,
        )
        condition_text.set_text("Current condition\n\n" + scene.condition_summary(state["condition"]))
        status_text.set_text(
            f"Method: {scene.METHOD_LABELS[state['method']]} | "
            "drag the 3D scene to rotate; use wheel to zoom"
        )
        figure.canvas.draw_idle()

    def on_random(_event) -> None:
        state["condition"] = scene.random_condition()
        redraw()

    def on_method(label: str) -> None:
        state["method"] = label_to_method[label]
        redraw()

    def on_save(_event) -> None:
        output = Path(__file__).with_name("haa_swing_range_ui_snapshot.png")
        figure.savefig(output, dpi=190)
        status_text.set_text(f"Saved: {output}")
        figure.canvas.draw_idle()

    def on_scroll(event) -> None:
        if event.inaxes is not ax:
            return
        _zoom_3d_axes(ax, 0.88 if event.button == "up" else 1.14)
        figure.canvas.draw_idle()

    random_button.on_clicked(on_random)
    method_radio.on_clicked(on_method)
    save_button.on_clicked(on_save)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    # Retain widget/callback references for Matplotlib's garbage collector.
    figure._haa_ui = {  # type: ignore[attr-defined]
        "scene": scene,
        "state": state,
        "random_button": random_button,
        "method_radio": method_radio,
        "save_button": save_button,
        "redraw": redraw,
    }
    redraw()
    if show:
        plt.show()
    return figure


def create_visualization_3d(
    network: HaaRangeNetwork,
    output: Path,
    *,
    device: torch.device,
    validation_mae: float,
    sweep_samples: int,
    max_triangles_per_link: int,
) -> None:
    """Render robot, envelope prism and all network-predicted HAA sweeps."""
    config = network.config
    condition = torch.tensor(DEMO_CONDITION, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        predicted = network(condition)[0].detach().cpu().numpy()

    values = condition_dict(config.condition_names, np.asarray(DEMO_CONDITION, dtype=np.float64))
    asset_cfg, init_state = load_el4090_config()
    joint_angles = build_prior_joint_angles(
        init_state["default_joint_angles"],
        init_state["mammal_default_joint_angles"],
        values,
    )
    urdf_path = resolve_asset_path(asset_cfg["file"])
    links, joints = parse_urdf(urdf_path)
    rng = np.random.RandomState(4090)
    robot_triangles = load_robot_triangles(links, max_triangles_per_link, rng)
    root_transform = transform_matrix(np.asarray(init_state["pos"], dtype=float), np.zeros(3))

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=24, azim=-132)

    envelope_points_3d = draw_envelope_prism(ax, values)
    draw_mammal_robot(
        ax,
        links,
        joints,
        robot_triangles,
        joint_angles,
        root_transform,
    )

    colors = ("#D62828", "#2A9D8F", "#F4A261", "#8E44AD", "#1D4ED8", "#7A5C00")
    sweep_points = []
    legend_handles = [
        Patch(facecolor="#6EC6FF", edgecolor="#0077B6", alpha=0.25, label="envelope"),
        Patch(facecolor="#777777", edgecolor="#333333", alpha=0.7, label="prior pose robot"),
    ]
    for leg_index, (leg_name, color) in enumerate(zip(config.leg_names, colors)):
        lower, upper = predicted[leg_index]
        chains = sample_leg_sweep(
            joints,
            joint_angles,
            root_transform,
            leg_name,
            float(lower),
            float(upper),
            sweep_samples,
        )
        draw_leg_sweep(ax, chains, color, leg_name, float(lower), float(upper))
        sweep_points.append(chains.reshape(-1, 3))
        legend_handles.append(Line2D([0], [0], color=color, linewidth=3, label=f"{leg_name} foot arc"))

    axis_support = np.array(((-0.95, -0.85, -0.55), (0.95, 0.85, 0.75)), dtype=float)
    set_axes_equal(
        ax,
        np.vstack((envelope_points_3d, *sweep_points, axis_support)),
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.85, 0.85)
    ax.set_zlim(-0.58, 0.78)
    ax.set_box_aspect((2.0, 1.7, 1.35))
    ax.set_title(
        "EL4090 3D envelope and network-generated HAA swing ranges\n"
        f"network validation MAE={validation_mae:.5f} rad; "
        "colored surfaces are leg sweeps, thick curves are foot trajectories",
        fontsize=12,
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)
    print(f"saved_3d_visualization={output}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Test and visualize the HAA range network.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("haa_swing_range_visualization.png"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--labels", choices=("analytic", "monte_carlo"), default="analytic")
    parser.add_argument("--mc-samples", type=int, default=4096)
    parser.add_argument("--plot-2d", action="store_true", help="Draw the earlier numerical diagnostic instead of 3D.")
    parser.add_argument("--interactive", action="store_true", help="Open the interactive 3D UI.")
    parser.add_argument(
        "--method",
        choices=("analytic", "monte_carlo", "network"),
        default="network",
        help="Initial HAA range method in interactive mode.",
    )
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--sweep-samples", type=int, default=31)
    parser.add_argument("--max-triangles-per-link", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.train_steps < 1:
        parser.error("--train-steps must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    network, validation_mae = obtain_network(
        device,
        checkpoint=args.checkpoint,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        label_method=args.labels,
        monte_carlo_samples=args.mc_samples,
    )
    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        network.save_checkpoint(args.save_checkpoint)
        print(f"saved_checkpoint={args.save_checkpoint}")
    if args.interactive:
        launch_interactive_ui(
            network,
            device=device,
            validation_mae=validation_mae,
            monte_carlo_samples=args.mc_samples,
            sweep_samples=args.sweep_samples,
            max_triangles_per_link=args.max_triangles_per_link,
            seed=args.seed,
            initial_method=args.method,
        )
    elif args.plot_2d:
        create_diagnostic_2d(
            network,
            args.output,
            device=device,
            validation_mae=validation_mae,
            monte_carlo_samples=args.mc_samples,
        )
    else:
        create_visualization_3d(
            network,
            args.output,
            device=device,
            validation_mae=validation_mae,
            sweep_samples=args.sweep_samples,
            max_triangles_per_link=args.max_triangles_per_link,
        )


if __name__ == "__main__":
    main()
