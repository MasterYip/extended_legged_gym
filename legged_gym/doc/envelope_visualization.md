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

Both scripts support three modes:

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
occupy unique sectors by default. The three sectors nearest each lateral axis
are always included so the left and right middle-leg workspaces are affected;
the remaining sectors vary with the seed. Returns use at most the first 12% of
their feasible radial annulus, while lateral anchors use at most 4.2%, placing
them close to the minimum collision-safe radius. Faces without a return retain
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
Compute-only motion: 120 frames; ... naive violations; 0 accepted violations; minimum scale ...
```

Naive violations are diagnostic: they show where direct interpolation inside the
joint box would exceed the prescribed envelope. The accepted pose is backtracked
toward the feasible anchor, so accepted violations must remain zero.

To check another deterministic cloud:

```bash
python legged_gym/scripts/visualize_lidar_free_envelope_gym.py \
  --compute_only --seed 4091 --point_count 20 --near_band_fraction 0.12
```

`--point_count` may be smaller than `--directions`; it must be positive.
`--near_band_fraction` is the maximum fraction of the feasible radial annulus
used by primary returns; reduce it toward zero to move points closer to the
minimum safe radius. The output reports point-constrained and reference-capped
face counts, the band fraction, and lateral anchor sectors.

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

## Evidence Location Policy

Generated PNG and JSON files must be stored outside the
`extended_legged_gym` Git repository. Both scripts reject screenshot paths inside
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
$(0,1]$; smaller values move returns closer to the robot without reducing the
declared robot clearance. Start from the defaults when testing a new machine,
then change one parameter at a time.

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
```
