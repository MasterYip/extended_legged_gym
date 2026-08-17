# Mathematical model

## Frame and capsule approximation

All planar quantities use the body-yaw frame: `+x` points forward, `+y` left,
and yaw is removed with the body origin fixed. Link meshes are approximated by
registered capsules. Capsule `l` has link-frame endpoints which forward
kinematics transforms to `a_l(q), b_l(q)` and radius `r_l`. This is an explicit
collision proxy, not a mesh-complete model.

For `K >= 3` evenly spaced unit normals

```text
u_k = (cos(theta_k), sin(theta_k)), theta_k = 2*pi*k/K,
```

the occupied support is

```text
h_k^occ(q) = max_l (max(u_k^T a_l(q), u_k^T b_l(q)) + r_l).
```

It induces the finite-normal outer polygon

```text
P_K(q) = {p in R^2 : u_k^T p <= h_k^occ(q) for every k}.
```

More directions reduce angular discretization error. They do not remove the
capsule approximation or convert the convex support model into exact collision
geometry.

## Allowed envelope, margin, and feasibility

An allowed polygon is represented by support `h^allowed` on the same normals.
For unit normals a metric margin `m` is exactly `h^allowed + m` for the
represented half-spaces. A pose is feasible when

```text
h_k^occ(q) <= h_k^allowed + tolerance for all k.
```

For viewer parameters, support is the maximum projection of the six declared
front/middle/back vertices, then the margin is added. A feasible reference is
chosen in order from a requested pose, the joint-box center, then the nearest
feasible deterministic candidate. If none exists, APIs return an explicit
infeasible marker rather than invented ranges.

## Rejection projections

Reference-pinned rejection sweeps joint `j` through its mechanical interval
while all other joints remain at a validated feasible reference `r`. Values are
rejected where `(q_j, r_-j)` violates the envelope. This is not existential over
the other joints: moving them may recover a rejected pinned value.

HAA rejection has three preserved modes:

- `pinned`: exact HFE/KFE pins are feasible; use the pinned HAA sweeps.
- `fold`: pins are infeasible but a six-HAA fold exists; reject a leg's HAA
  value only when no sampled configuration of the other five HAA joints fits.
- `none`: no sampled HAA tuple fits; report every HAA mechanical range rejected.

Per-leg 3-DOF rejection similarly treats one leg's three joints as a free box,
with the other legs pinned at a feasible reference. A value of one joint is
rejected only when no sampled setting of that leg's other two joints fits.
This is existential within a leg, not over all 18 joints.

## Approximation and conservative bias

Existential projections use deterministic scrambled Sobol samples (default seed
4090), project the feasible cloud into bins, and complement observed accessible
bins. A thin accessible set can be missed, biasing the result toward rejection.
`min_rej_span` removes small sampling-noise bands. The tuned fold defaults are
8192 samples, 257 bins, and a 0.15 rad minimum rejected span; fold feasibility
uses every other direction for a 32-to-16 normal reduction. These defaults
preserve the accepted band structure and measured <=50 ms fold-core target on
the recorded development CPU, not a hardware-independent deadline.

Tolerance is applied only to the support inequality. Sampled range exports and
zero observed validation violations are evidence for the registered samples;
they are not global certificates.
