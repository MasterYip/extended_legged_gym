# Troubleshooting

## IsaacGym: `libpython3.8.so.1.0: cannot open shared object file`

**Error:**
```
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

**Root Cause:**

Conda intentionally does not modify `LD_LIBRARY_PATH` — it relies on RPATH embedded in compiled binaries. `gym_38.so` was compiled by NVIDIA without an RPATH pointing to the conda env's lib directory.

On **Ubuntu 20.04**, Python 3.8 is a system library (`/usr/lib/x86_64-linux-gnu/libpython3.8.so.1.0`), so the dynamic linker found it automatically. On **Ubuntu 22.04+**, the system ships Python 3.10/3.11 and Python 3.8 only lives inside the conda env — the linker never searches there unless told to.

**Fix:**

Add the conda env's lib path to `LD_LIBRARY_PATH` on activation:

```bash
mkdir -p ~/miniforge3/envs/isaacgym/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
  >> ~/miniforge3/envs/isaacgym/etc/conda/activate.d/env_vars.sh
```

Re-activate the environment for the change to take effect:

```bash
conda activate isaacgym
```

Using `$CONDA_PREFIX` keeps the script portable across machines.

---

## Rollout Envs: `PxgAABBManager` Warning & Slow `step_rollout`

**Warning:**
```
/buildAgent/work/.../PxgAABBManager.cpp (1048) : invalid parameter :
The application needs to increase PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity
to 779948463, otherwise, the simulation will miss interactions
```

**Observed Behaviour:**

- Fewer rollout environments → faster `step_rollout` (suggesting `main_env × rollout_envs ≈ const`).
- Setting 0 env spacing causes robots to crowd together, triggering the broadphase capacity warning.

**References:**

- [NVIDIA Forums: Issue with environment spacing and PxgDynamicsMemoryConfig](https://forums.developer.nvidia.com/t/issue-with-environment-spacing-and-pxgdynamicsmemoryconfig-foundlostaggregatepairscapacity/198272/9)

**TODO:** For rollout envs, try setting slightly different spawn positions to avoid overcrowding.
