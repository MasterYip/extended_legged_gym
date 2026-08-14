# Batched Kinematic Support Envelopes for EL4090

## Scope and coordinate convention

This module adds and proposes an auditable, differentiable geometry layer
alongside the hand-authored symmetric hexagon. It is not yet wired into the
training environment or deployed policy path. All planar quantities are
expressed in the
EL4090 body-yaw frame: \(+x\) points forward and \(+y\) points left. Two sets
must remain conceptually separate:

1. the **occupied-body envelope**, approximated by 19 link capsules; and
2. the **reachable-foot envelope**, estimated from forward-kinematic foot
   samples over an explicitly declared joint domain.

The policy interface remains unchanged. Geometry is projected to the legacy
condition vector \([B,8]\), HAA intervals are exported as \([B,6,2]\), and the
envelope-v2 observation remains \([B,83]\). This is interface compatibility,
not a claim that the old symmetric parameterization preserves all geometry.

The publication figure and its drawing source are paper-owned artifacts and
belong in the EL4090 paper repository. This RL repository intentionally keeps
only the geometry implementation, tests, benchmark, and technical evidence.

## Method

For a URDF joint \(j\), the link transform is evaluated with Torch as

\[
{}^0T_j(q) = {}^0T_{p(j)}\,T_{p(j),j}^{\mathrm{URDF}}\,
\begin{bmatrix}R(\hat a_j,q_j)&0\\0&1\end{bmatrix}.
\]

The fixed joint graph is traversed once in topological order. Tensor operations
carry every leading batch dimension; there is no Python loop over environments.
URDF limits can be contracted around their center by a declared soft fraction.
For the current EL4090 asset, the raw revolute bounds are \([-3,3]\) rad and a
soft fraction of \(0.9\) gives \([-2.7,2.7]\) rad.

For capsule \(l\), with transformed endpoints \(a_l(q),b_l(q)\), radius
\(r_l\), and unit planar normal \(u_k\), the occupied support is

\[
h_k^{\mathrm{occ}}(q)=\max_l\left[
\max\{u_k^\top a_l(q),u_k^\top b_l(q)\}+r_l\right].
\]

The corresponding fixed-direction half-space model is

\[
\mathcal E(q)=\{p\in\mathbb R^2\mid u_k^\top p\le h_k(q),
\; k=1,\ldots,K\}.
\]

Because every \(u_k\) is normalized, a metric safety margin \(m\) is exact for
the represented half spaces:

\[
h_k^{\mathrm{safe}}=h_k^{\mathrm{occ}}+m.
\]

The finite-direction intersection outer-approximates the capsule convex hull;
increasing \(K\) reduces directional discretization error. Reachable-foot
support uses the same directions but maximizes sampled FK foot positions:

\[
h_k^{\mathrm{foot}}=\max_{q\in Q_s,\ell\in\mathcal L}
u_k^\top f_\ell(q).
\]

These two support vectors must not be substituted for one another: one models
occupied structure, the other sampled reachability.

## Joint-range export and diagnostics

`export_envelope_joint_ranges` evaluates capsule support for candidate tensors
\([\ldots,S,J]\) against a requested allowed support \([\ldots,K]\), derives a
box only from candidates satisfying every support inequality, and separately
evaluates validation tensors \([\ldots,V,J]\). Diagnostics report candidate and
validation feasible counts, false exclusions of feasible validation samples,
and violations among independent Sobol samples drawn from the exported
Cartesian box. The label **conservative on registered box-validation samples**
is emitted only when every batch has feasible candidates and the sampled box
has zero envelope violations; otherwise it is **approximate**. An empty result
has `valid=False` and NaN ranges rather than invented limits.

`export_sample_bounding_ranges` remains available as a clearly named low-level
helper when feasibility was established externally. It must not be interpreted
as envelope-conditioned certification.

This wording is deliberate. An axis-aligned range covers candidate feasible
samples but does not prove every Cartesian combination inside the box feasible.
Nor does sample validation establish global conservatism. A stronger claim
requires interval methods, optimization certificates, or exhaustive bounds.

HAA output defaults to simulator leg order
`LB LF LM RB RF RM`. The legacy checkpoint order
`RF RM RB LF LM LB` is available explicitly through `LEGACY_HAA_ORDER`; silent
ordering assumptions are avoided.

## Policy compatibility

`legacy_condition_from_support` reads cardinal support values, projects them to
the legacy symmetric fields `(wf, wm, wb, xf, xb)`, clamps those fields to the
training ranges, and appends three caller-supplied morphology priors. The width
collapse is intentionally approximate because asymmetric support cannot be
encoded losslessly in five symmetric fields.

`append_legacy_envelop2_observation` preserves the deployed 83-D contract:
66 proprioceptive values, three morphology priors, six HAA centers, six HAA
half-ranges, and sine/cosine gait phase. It performs assembly only; it does not
alter the trained policy or claim checkpoint equivalence beyond shape/order.

## Complexity

Let \(B\) be batch size, \(J\) URDF joints, \(L\) capsule count, \(P=2\)
capsule endpoints, and \(K\) support directions. FK costs \(O(BJ)\); capsule
projection costs \(O(BLPK)\). The dominant materialized projection has
\(O(BLPK)\) memory. With EL4090 defaults \(J=18\), \(L=19\), and \(K=64\),
the method is linear in environment count and direction count.

## Verification

The deterministic test suite checks:

- float64 Torch FK against an independent NumPy transform oracle at default and
  mammal poses;
- capsule endpoint containment, dtype and batch shape, and effective limits;
- exact fixed-normal margin offsets and point queries;
- autograd against centered finite differences away from support ties;
- deterministic Sobol samples and reachable-foot support shape;
- conservative/approximate range diagnostics;
- envelope-conditioned narrowing and empty-range behavior, effective limits,
  batched allowed support, and Cartesian-box envelope violations;
- `[B,8]`, `[B,6,2]`, and `[B,83]` contracts plus simulator/legacy remapping.

Run from `legged_gym/legged_gym`:

```bash
/home/user/miniforge3/envs/isaacgym/bin/python tests/test_kinematic_envelope.py
```

The revised recorded run passed 10/10 tests. The symmetry golden check has a
5 um tolerance and the float32/float64 support check has a 2e-6 m tolerance.
CUDA parity was not executed in
this task because the project GPU was reserved by `RL-SMOKE-002`; no CUDA claim
is made here.

## CPU benchmark

The retained batch-2048 benchmark uses Torch float32, 64 directions, 19
capsules, 16 CPU threads, three warmups, and ten timed iterations:

```bash
/home/user/miniforge3/envs/isaacgym/bin/python \
  utils/envelop/benchmark_kinematic_envelope.py \
  --output utils/envelop/figures/kinematic_envelope_cpu_benchmark.json
```

The median was **25.289 ms** (minimum 23.203 ms), equivalent to **80,984
environment evaluations/s**. Output shape was `[2048,64]`. Process peak RSS was
433,996 KiB; measured peak increase after allocation of inputs was 65,212 KiB.
Timing and RSS are machine- and allocator-dependent, so the JSON artifact is
evidence for this run rather than a cross-platform guarantee.

## Publication figure ownership

Graph-drawing scripts and rendered publication assets are maintained in the
EL4090 paper repository rather than `extended_legged_gym`. They should consume
the equations, coordinate convention, limitations, and recorded benchmark in
this note without duplicating the executable geometry implementation.

## Isaac Gym comparison demo

`scripts/visualize_kinematic_envelope_gym.py` is an RL-owned simulator demo,
not a publication graph generator and not part of the training environment. It
loads three fixed-base EL4090 actors without a policy checkpoint and compares
compact-mammal, nominal-spider, and wide-low presets. Each actor carries its
occupied capsule boundary, sampled reachable-foot boundary, and six physical
hip-centered HAA interval arcs with bound rays and current-angle markers. The
actors move through smooth deterministic paths inside their own exported
18-joint intervals; the occupied boundaries and HAA markers update from those
same poses on every rendered frame.

The viewer prints the exact direction vectors, occupied/allowed/reachable
support values, current 18-joint pose, range diagnostics, and HAA intervals in
simulator order `LB LF LM RB RF RM`. Controls are printed at launch: number keys
select a preset; Space cycles; A toggles automatic selection; M pauses or
resumes joint and envelope motion; X resets the motion phase; O, R, and H
toggle the three geometry layers; C cycles camera modes; P captures; and Esc
exits.

From `legged_gym/legged_gym`, validate preset computation without a viewer:

```bash
LD_LIBRARY_PATH=/home/user/miniforge3/envs/isaacgym/lib:$LD_LIBRARY_PATH \
/home/user/miniforge3/envs/isaacgym/bin/python \
  scripts/visualize_kinematic_envelope_gym.py --compute_only
```

Run a bounded GPU-0 viewer smoke and write one screenshot plus matching JSON
outside this repository:

```bash
LD_LIBRARY_PATH=/home/user/miniforge3/envs/isaacgym/lib:$LD_LIBRARY_PATH \
/home/user/miniforge3/envs/isaacgym/bin/python \
  scripts/visualize_kinematic_envelope_gym.py \
  --compute_device_id 0 --graphics_device_id 0 \
  --max_steps 90 --auto_cycle_steps 20 --motion_period_steps 60 \
  --screenshot_step 89 \
  --screenshot /tmp/ENV-DESIGN-003-motion/isaac_gym_envelope_motion.png
```

The paired JSON is compact run evidence: it records visited presets, frame and
joint-sample counts, exported bounds, observed per-joint extrema, violation
count, and maximum bound excess. The viewer fails immediately if a rendered
pose exceeds its active exported interval.

Generated captures and numeric evidence are task-owned artifacts. Keep them in
the task record or another external results location, never in the
`extended_legged_gym` Git repository.

## LiDAR-derived point-free envelope demo

`scripts/visualize_lidar_free_envelope_gym.py` demonstrates the complete
LiDAR-to-motion contract on one real, fixed-base EL4090 model. A deterministic
synthetic scan provides full angular-sector coverage, three near-return
clusters, and two farther gaps. Arguments expose the seed, return count, radius
bounds, minimum robot clearance, and prescribed point clearance.

Let normalized fixed normals be $u_k$, and assign each return $p_i$ to its
nearest angular sector $s(i)$. The example declares the restricted polygon
family

$$
\mathcal P(h)=\{x\in\mathbb R^2\mid u_k^\top x\le h_k,\ k=1,\ldots,K\},
$$

with the separable safety contract

$$
u_{s(i)}^\top p_i-h_{s(i)}\ge d_{\mathrm{point}}.
$$

Its coordinatewise maximum is

$$
h_k^\star=\min_{i:s(i)=k}u_k^\top p_i-d_{\mathrm{point}}.
$$

Every sector has an active limiting return. Increasing any one
$h_k^\star$ violates that return's clearance. This is the implemented and
tested maximality claim; it is not a claim of a global maximum over arbitrary
polygon topologies or alternative point-to-face assignments.

The support $h^\star$ is passed to the sampled 18-joint range export. An
axis-aligned joint box alone is not a collision certificate, so the animation
proposes a smooth box-bounded pose and backtracks it toward a known feasible
anchor until both constraints hold:

$$
q^-\le q(t)\le q^+,
\qquad
h_k^{\mathrm{occ}}(q(t))\le h_k^\star\quad\forall k.
$$

Light cyan means the prescribed LiDAR-derived envelope; dark teal means the
current occupied capsule envelope. White crosses are returns, cyan spokes mark
active limiting clearances, amber shows physical HAA ranges, and muted blue is
the optional reachable-foot layer. Red is reserved for a true violation.
Controls are printed at launch: `G` regenerates with the next seed; `M`
pauses motion; `X` resets phase; `L`, `P`, `O`, `H`, and `R` toggle
layers; `C` changes camera; `S` captures; and Esc exits.

Validate without a viewer:

```bash
LD_LIBRARY_PATH=/home/user/miniforge3/envs/isaacgym/lib:$LD_LIBRARY_PATH \
/home/user/miniforge3/envs/isaacgym/bin/python \
  scripts/visualize_lidar_free_envelope_gym.py \
  --compute_only --seed 4090 --point_count 192 --directions 48
```

Run the bounded GPU viewer, keeping generated output outside this repository:

```bash
mkdir -p /tmp/ENV-DESIGN-003-lidar
LD_LIBRARY_PATH=/home/user/miniforge3/envs/isaacgym/lib:$LD_LIBRARY_PATH \
/home/user/miniforge3/envs/isaacgym/bin/python \
  scripts/visualize_lidar_free_envelope_gym.py \
  --compute_device_id 0 --graphics_device_id 0 \
  --seed 4090 --point_count 192 --directions 48 \
  --max_steps 180 --motion_period_steps 120 --screenshot_step 179 \
  --screenshot /tmp/ENV-DESIGN-003-lidar/lidar_free_envelope.png
```

The matching JSON records scan parameters, active limiting returns, prescribed
support and vertices, exported ranges for all 18 joints, visible-layer
semantics, and motion compliance. The viewer aborts if an accepted frame
exceeds either its joint interval or occupied-envelope support.

## Limitations

- Capsules are explicit low-cost proxies calibrated from URDF joint spans; they
  are not mesh-complete collision geometry.
- Fixed support directions approximate curved boundaries between normals.
- Reachability and joint-range labels are sample-defined and depend on the
  declared domain, seed, and validation set.
- Support maxima are differentiable almost everywhere; gradients are
  non-unique at ties where the active endpoint, capsule, or link changes.
- The legacy 8-D projection discards asymmetry and is bounded to historical
  training ranges. It exists for checkpoint compatibility, not as a new state
  representation recommendation.
- The viewer is a fixed-base geometry demonstration, not a policy rollout,
  checkpoint evaluation, GPU parity result, or hardware safety validation.
- The module is a proposed utility and has not been integrated into the
  EL4090 environment, observation computation, training loop, or deployment.
