# EL4090 Envelope Visualization Examples

This guide covers the standalone Isaac Gym examples for inspecting the EL4090
kinematic envelope and the LiDAR-prescribed collision-free envelope. The examples
use the EL4090 URDF and deterministic Torch geometry directly; they do not load an
RL policy or checkpoint.

## Prerequisites

- NVIDIA Isaac Gym Preview 4 is installed and importable from the `isaacgym`
  Conda environment.
- The EL4090 URDF is available at
  `resources/robots/el_4090/urdf/el_4090.urdf`.
- An NVIDIA GPU and a working graphical display are required for the viewer.
  Compute-only validation does not open a window.
- The commands below are run from the `legged_gym` project directory:

```bash
cd /home/user/CodeSpace/Python/PredictiveDiffusionPlanner_Dev/extended_legged_gym/legged_gym
conda activate isaacgym
```

If Isaac Gym cannot locate `libpython3.8.so.1.0`, expose the active environment's
library directory before running an example:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Example Overview

| Example | Script | Purpose |
| --- | --- | --- |
| Kinematic comparison | `legged_gym/scripts/visualize_kinematic_envelope_gym.py` | Compare compact, nominal, and wide EL4090 envelopes while all joints move within their exported intervals. |
| LiDAR free envelope | `legged_gym/scripts/visualize_lidar_free_envelope_gym.py` | Generate a structured synthetic 2D LiDAR cloud, derive the point-free envelope, export joint intervals, and animate a constraint-compliant pose. |
| ZhangHT border sliders | `legged_gym/scripts/visualize_legacy_slider_envelope_gym.py` | Replace random LiDAR returns with slider-border ray intersections, then run the unchanged maximum-envelope and joint-range pipeline. |

All three scripts support compute-only, bounded, and interactive modes; the slider example additionally opens a Tkinter control panel. The original two viewers support three modes:

1. `--compute_only` validates geometry without creating Isaac Gym.
2. `--max_steps N` runs a bounded viewer and exits naturally after `N` frames.
3. Omitting both options starts an interactive viewer that runs until `Esc` or
   the window close button is used.

## Kinematic Envelope Comparison

### Compute-only validation

```bash
python legged_gym/scripts/visualize_kinematic_envelope_gym.py --compute_only
```

The command prints the exact preset definitions and their range-export
diagnostics. A successful run ends with:

```text
Compute-only validation complete; no simulator or policy checkpoint used.
```

### Bounded viewer and evidence capture

Create the output directory outside the RL repository, then run:

```bash
mkdir -p /tmp/env-design-003
python legged_gym/scripts/visualize_kinematic_envelope_gym.py \
  --max_steps 240 \
  --motion_period_steps 240 \
  --screenshot /tmp/env-design-003/kinematic_envelope.png \
  --screenshot_step 180
```

The viewer writes the PNG and a matching
`/tmp/env-design-003/kinematic_envelope.json`. The terminal acceptance summary
must report zero range violations and zero maximum bound excess:

```text
Motion compliance: ... joint samples, 0 violations, max excess 0 rad
```

### Interactive viewer

```bash
python legged_gym/scripts/visualize_kinematic_envelope_gym.py \
  --screenshot /tmp/env-design-003/kinematic_envelope.png
```

The initial automatic cycle changes the selected preset every 180 steps. Use
`--no_auto_cycle` to keep the initial selection, `--auto_cycle_steps N` to change
the interval, `--no_motion` to start paused, and `--directions N` to change the
support discretization.

| Key | Action |
| --- | --- |
| `1`, `2`, `3` | Select compact, nominal, or wide preset. |
| `Space` | Select the next preset. |
| `A` | Toggle automatic preset cycling. |
| `M` | Pause or resume joint and envelope motion. |
| `X` | Reset the deterministic motion phase. |
| `O` | Toggle the occupied capsule boundary. |
| `R` | Toggle the reachable-foot boundary. |
| `H` | Toggle the six HAA interval arcs and current markers. |
| `C` | Cycle overview, top, and selected-preset cameras. |
| `P` | Write the configured screenshot and matching JSON evidence. |
| `Esc` | Exit. |

### Kinematic-view colors

| Color | Meaning |
| --- | --- |
| Teal | Current occupied capsule boundary. |
| Cyan | Preset reachable-foot boundary. |
| Amber | Exported HAA interval arcs. |
| Red radial marker | Current HAA direction, derived from the physical hip-to-foot body-XY vector at the current pose. |
| Red, amber, or cyan robot accents | Compact, nominal, or wide preset identity; these accents are not violation indicators. |

## LiDAR-Prescribed Free Envelope

The LiDAR example creates 20 sparse seeded returns inside the pre-obstacle
reachable envelope and outside the baseline occupied robot envelope. Returns
occupy unique sectors by default. The five sectors nearest each lateral axis
are always included, so 10 of 20 returns directly constrain the left and right
middle-leg workspaces. The remaining sectors vary with the seed. Returns use at
most the first 5% of their feasible radial annulus, while lateral anchors use at
most 1.75%. Lateral anchors require 0.025 m baseline clearance versus 0.05 m
for other returns. With 0.02 m point clearance, this leaves approximately
0.005 m between the baseline occupied support and a limiting lateral envelope
face. Faces without a return retain
the blue pre-obstacle reference cap. Seed changes randomize the other
constrained faces, cluster and gap locations, angular jitter, radial placement,
and therefore the light-cyan polygon shape. For direction $\mathbf u_k$, the
accepted animation maintains

$$
h_{\mathrm{occ}}(\mathbf u_k;\mathbf q)
\le h_{\mathrm{free}}(\mathbf u_k),
\qquad k=1,\ldots,K,
$$

and every animated joint remains inside its exported interval
$q_j\in[q_j^-,q_j^+]$.

### Compute-only validation

```bash
python legged_gym/scripts/visualize_lidar_free_envelope_gym.py --compute_only
```

The default run reports the seed, randomized point-cloud structure, clearances,
candidate reduction, joint-interval shrinkage, and a complete motion sweep. The
acceptance line must report zero accepted violations:

```text
Compute-only motion: 120 frames; ... naive violations; 0 accepted violations; fixed scale ...; maximum cyclic joint step ...
```

Naive violations are diagnostic: they show where direct interpolation inside the
joint box would exceed the prescribed envelope. The complete cyclic trajectory is backtracked once with a single fixed scale
toward the feasible anchor, so accepted violations must remain zero.

To check another deterministic cloud:

```bash
python legged_gym/scripts/visualize_lidar_free_envelope_gym.py \
  --compute_only --seed 4091 --point_count 20 \
  --near_band_fraction 0.05 --lateral_robot_clearance 0.025 \
  --lateral_anchors_per_side 5
```

`--point_count` may be smaller than `--directions`; it must be positive.
`--near_band_fraction` is the maximum fraction of the feasible radial annulus
used by primary returns; reduce it toward zero to move points closer to the
minimum safe radius. `--lateral_robot_clearance` controls the lateral minimum
distance independently and must be greater than `--point_clearance` and no
greater than `--robot_clearance`. The output reports point-constrained and
reference-capped face counts, band fraction, and lateral anchor sectors.

### Bounded viewer and evidence capture

```bash
mkdir -p /tmp/env-design-003
python legged_gym/scripts/visualize_lidar_free_envelope_gym.py \
  --seed 4090 \
  --max_steps 180 \
  --motion_period_steps 120 \
  --screenshot /tmp/env-design-003/lidar_free_envelope.png \
  --screenshot_step 150
```

The matching JSON records the cloud seed and structure, clearances, supports,
exported joint limits, range impact, visible layers, and motion compliance. A
successful bounded run ends with zero joint and envelope violations and zero
maximum support excess:

```text
Accepted compliance: ... joint samples; 0 joint violations; 0 envelope violations; max support excess 0 m
```

### Interactive viewer

```bash
python legged_gym/scripts/visualize_lidar_free_envelope_gym.py \
  --screenshot /tmp/env-design-003/lidar_free_envelope.png
```

| Key | Action |
| --- | --- |
| `G` | Regenerate the cloud and envelope with `seed + 1`; motion and statistics reset. |
| `M` | Pause or resume feasible motion. |
| `X` | Reset the deterministic motion phase. |
| `L` | Toggle white LiDAR returns and clearance spokes. |
| `P` | Toggle the light-cyan prescribed free envelope. |
| `O` | Toggle the dark-teal current occupied envelope. |
| `H` | Toggle the amber HAA ranges and current markers. |
| `R` | Toggle the blue pre-obstacle reachable reference. |
| `C` | Cycle overview and top cameras. |
| `S` | Write the configured screenshot and matching JSON evidence. |
| `Esc` | Exit. |

### LiDAR-view colors

| Color | Meaning | Required relation |
| --- | --- | --- |
| White | Synthetic 2D LiDAR returns. | Inside the blue reference and outside the baseline robot envelope. |
| Blue | Pre-obstacle unconstrained reachable-foot reference. | Outer reference used to bound point generation and the prescribed envelope. |
| Light cyan | Prescribed point-free envelope and active clearance spokes. | Must not exceed the blue reference. |
| Dark teal | Current occupied capsule envelope. | Must remain inside the light-cyan envelope. |
| Amber | Exported HAA intervals and current body-XY hip-to-foot directions from URDF FK. | Motion must remain inside the exported intervals. |
| Red | Actual joint or envelope violation only. | Must never appear in an accepted run. |

Every displayed line uses two vertically offset strokes, matching
`visualize_kinematic_envelope_gym.py`. This applies consistently to envelope
boundaries, LiDAR targets, clearance spokes, HAA arcs, bounds, and markers.

The blue and dark-teal boundaries describe different quantities: blue is a
pre-obstacle reachable-foot reference, while dark teal is the robot's current
occupied capsule support. The light-cyan boundary is the active collision-free
constraint between them.

## ZhangHT Legacy-Border Slider

This example is the existing LiDAR envelope demo with one controlled change:
random returns are replaced by points derived from ZhangHT's five-parameter
motion generator, geometric validity checks, and viewer layers remain the same.
The random-cloud demo's optional material-impact gates are not used to reject a
valid non-binding slider border.

Let the registered outward directions be $u_k$, and let
$\partial\mathcal B_Z(\theta)$ be the ZhangHT border for slider vector
$\theta=(w_f,w_m,w_b,x_f,x_b)$. The raw return distance is the first positive
ray-border intersection

$$
\rho_k^Z=\min\{\rho>0\mid \rho u_k\in\partial\mathcal B_Z(\theta)\}.
$$

Returns retain the LiDAR viewer's reachable-reference invariant. For reference
support values $h_j^{\mathrm{ref}}$ and the existing containment margin
$\varepsilon=0.005\,\mathrm m$, the radial cap is

$$
\rho_k^{\mathrm{ref}}
=\min_{j:\,u_j^\top u_k>0}
\frac{h_j^{\mathrm{ref}}-\varepsilon}{u_j^\top u_k},
\qquad
p_k=\min(\rho_k^Z,\rho_k^{\mathrm{ref}})u_k.
$$

Only $p_k$ changes with the sliders. The original fixed-normal LiDAR optimizer
is called without modification:

$$
h_k^\star
=\min\left(h_k^{\mathrm{ref}},\;u_k^\top p_k-d_{\mathrm{point}}\right),
\qquad d_{\mathrm{point}}=0.02\,\mathrm m.
$$

The prescribed point-free envelope remains

$$
\mathcal E^\star=\{x\in\mathbb R^2\mid u_k^\top x\le h_k^\star,\ \forall k\}.
$$

It is the same coordinatewise maximum in the declared fixed-normal capped
polygon family used by `visualize_lidar_free_envelope_gym.py`. Because it is an
intersection of half-spaces, it is one connected convex polygon. There is no
rear/front decomposition. The dark-teal current envelope is likewise the
single existing capsule-support polygon, not a foot hull.

The unchanged exporter tests registered candidate configurations against
$\mathcal E^\star$, derives all 18 axis-aligned joint intervals, and validates
sampled combinations from the exported box. The viewer then uses the same
trajectory-wide feasible motion scale as the LiDAR example. A large slider
border may be clipped by the reachable reference, so a slider change can be
valid while leaving some exported intervals unchanged.

### Launch

Compute without opening either window:

```bash
python legged_gym/scripts/visualize_legacy_slider_envelope_gym.py --compute_only
```

Start the interactive Tkinter panel and Isaac Gym viewer:

```bash
python legged_gym/scripts/visualize_legacy_slider_envelope_gym.py \
  --screenshot /tmp/env-design-003/legacy_slider.png
```

Drag a slider to enqueue a recomputation. The 80 ms debounce coalesces dense
drag events, while mouse release applies the latest values immediately.
`Reset midpoint` selects $(0.45,0.50,0.45,0.75,-0.75)$ and `Maximum border`
selects $(0.60,0.70,0.60,0.90,-0.90)$. `Capture` writes the configured PNG and
matching JSON outside this repository. Invalid settings retain the last valid
result and show the reason in the Tk status line.

| Slider | Range [m] | Border coordinate |
| --- | ---: | --- |
| `front_width` | 0.30 to 0.60 | Lateral half-width at $x=x_f$. |
| `middle_width` | 0.30 to 0.70 | Lateral half-width at $x=0$. |
| `back_width` | 0.30 to 0.60 | Lateral half-width at $x=x_b$. |
| `forward_limit` | 0.60 to 0.90 | Front longitudinal coordinate $x_f$. |
| `backward_limit` | -0.90 to -0.60 | Rear longitudinal coordinate $x_b$. |

| Color | Meaning |
| --- | --- |
| White | ZhangHT border and its one-return-per-sector LiDAR samples. |
| Blue | Pre-obstacle reachable reference cap. |
| Light cyan | Single computed maximum point-free envelope $\mathcal E^\star$. |
| Dark teal | Single current robot capsule-support envelope. |
| Amber | Exported HAA intervals and current URDF hip-to-foot directions. |
| Red | Actual accepted joint or capsule-envelope violation only. |

For a deterministic callback smoke, use `--max_steps 120 --auto_sweep_steps 30`.
`--directions`, `--point_clearance`, `--candidate_count`,
`--validation_count`, and `--box_validation_samples` have the same meanings as
in the LiDAR example. `--motion_period_steps` controls the shared smooth cyclic
trajectory.

## Evidence Location Policy

Generated PNG and JSON files must be stored outside the
`extended_legged_gym` Git repository. All three scripts reject screenshot paths inside
the repository. Use `/tmp/env-design-003/` for temporary inspection or copy a
selected, reviewed result to the canonical task evidence directory managed by the
agent-team workflow. Do not place generated evidence in `legged_gym/doc/imgs`,
`legged_gym/logs`, or any other path under this repository.

The screenshot hotkeys (`P` for the kinematic comparison and `S` for the LiDAR
example) write evidence only when `--screenshot` was supplied at launch. Without
that argument, the hotkey has no output target.

## Troubleshooting

### `libpython3.8.so.1.0` is missing

Activate the `isaacgym` environment and set `LD_LIBRARY_PATH` as shown in
[Prerequisites](#prerequisites). Confirm the active interpreter with:

```bash
which python
python --version
```

### The viewer cannot be created

Check that `DISPLAY` names a reachable graphical session and that the selected
graphics device is valid:

```bash
echo "$DISPLAY"
nvidia-smi
```

On a multi-GPU system, select devices explicitly with
`--compute_device_id N --graphics_device_id N`. For a headless host, use
`--compute_only`; these scripts do not create an off-screen viewer.

### Evidence capture is rejected

Choose an absolute or relative path that resolves outside the RL repository, for
example `/tmp/env-design-003/example.png`. The JSON path is derived by replacing
the screenshot suffix with `.json`.

### LiDAR argument validation fails

Keep `--directions` at least 8, `--point_count` positive,
`--point_clearance` smaller than `--robot_clearance`, and
`--reference_containment_margin` positive. Keep `--near_band_fraction` in
$(0,1]$. Require
$d_{\mathrm{point}}<d_{\mathrm{lateral}}\le d_{\mathrm{robot}}$, and keep
`--lateral_anchors_per_side` positive. Start from the defaults when testing a
new machine, then change one parameter at a time.

### LiDAR materiality or feasibility fails

The script intentionally rejects a cloud if it does not remove at least
`--min_candidate_reduction_fraction` of the sampled candidates, shrink at least
one joint interval by `--min_joint_shrink_rad`, contain the baseline anchor, or
fit inside the pre-obstacle reference. Restore the default thresholds and cloud
parameters, or choose another explicit `--seed`; do not treat a rejected run as
valid evidence.

### The viewer stops unexpectedly

A positive `--max_steps` requests bounded execution and is expected to close the
viewer naturally. Use the default value `0` for an interactive session.

## Command Reference

Use the scripts' built-in help as the authoritative option list:

```bash
python legged_gym/scripts/visualize_kinematic_envelope_gym.py --help
python legged_gym/scripts/visualize_lidar_free_envelope_gym.py --help
python legged_gym/scripts/visualize_legacy_slider_envelope_gym.py --help
```
