# Isaac Gym Environments for Legged Robots

This repository contains Isaac Gym environments for training legged robots using reinforcement learning.

## Quick Start

```bash
conda activate pdplanner
```

## Robot Tasks Overview

### Anymal Robot Tasks
| Task Category | Description | Documentation |
|---------------|-------------|---------------|
| Batch Rollout | Trajectory optimization with batch rollout capability | [Details](doc/anymal_tasks.md#batchrolloutanymal) |
| Base Pose Adapt | Base pose adaptation for collision avoidance | [Details](doc/anymal_tasks.md#baseposeadapt-anymalc) |
| Flat Terrain | Basic locomotion on flat terrain (~200 epochs) | [Details](doc/anymal_tasks.md#anymalc-flat-terrain) |
| Load Adaptation | Load adaptation training | [Details](doc/anymal_tasks.md#loadadapt-anymalc-flat) |
| Pose Control | Pose control tasks | [Details](doc/anymal_tasks.md#pose-anymalc-flat) |
| Standing | Standing behavior training | [Details](doc/anymal_tasks.md#stand-anymalc-flat) |
| Rough Terrain | Locomotion on rough terrain (~200 epochs) | [Details](doc/anymal_tasks.md#anymalc-rough-terrain) |
| Trajectory Sampling | Gradient sampling for trajectory optimization | [Details](doc/anymal_tasks.md#anymalc-trajectory-gradient-sampling) |

### ElSpider Air Robot Tasks
| Task Category | Description | Documentation |
|---------------|-------------|---------------|
| Base Pose Adapt | Base pose adaptation for collision avoidance | [Details](doc/elspider_air_tasks.md#baseposeadapt-elspiderair) |
| Flat Terrain | Basic locomotion on flat terrain (~300-500 epochs) | [Details](doc/elspider_air_tasks.md#elspiderair-flat-terrain) |
| Batch Rollout | Trajectory optimization with batch rollout | [Details](doc/elspider_air_tasks.md#elspiderair-batch-rollout) |
| Batch Rollout Flat | Batch rollout on flat terrain (no perception) | [Details](doc/elspider_air_tasks.md#elspiderair-batch-rollout-flat) |
| Trajectory Sampling | Gradient sampling for trajectory optimization | [Details](doc/elspider_air_tasks.md#elspiderair-trajectory-gradient-sampling) |
| Pose Control | Pose control on flat terrain | [Details](doc/elspider_air_tasks.md#pose-elspiderair-flat) |
| Foot Tracking | Foot tracking in hang/ground modes | [Details](doc/elspider_air_tasks.md#foottrack-elspiderair) |
| Rough Terrain | Locomotion on rough terrain (~500 epochs) | [Details](doc/elspider_air_tasks.md#elspiderair-rough-terrain) |
| Rough RayCast | Rough terrain with raycast perception | [Details](doc/elspider_air_tasks.md#elspiderair-rough-raycast) |

### CyberDog2 Robot Tasks
| Task Category | Description | Documentation |
|---------------|-------------|---------------|
| Standing | Standing behavior training | [Details](doc/cyberdog2_tasks.md#cyberdog2-stand) |

## Detailed Documentation

- **[Anymal Tasks](doc/anymal_tasks.md)** - Complete guide for all Anymal robot training tasks
- **[ElSpider Air Tasks](doc/elspider_air_tasks.md)** - Complete guide for all ElSpider Air robot training tasks  
- **[CyberDog2 Tasks](doc/cyberdog2_tasks.md)** - Complete guide for CyberDog2 robot training tasks
- **[Configuration & Technical Notes](doc/configuration.md)** - Setup instructions, known issues, and technical details

## Common Command Patterns

| Command Type | Pattern |
|--------------|---------|
| Training | `python legged_gym/scripts/train.py --task=<task> --num_envs=<envs> --resume --headless` |
| Testing/Play | `python legged_gym/scripts/play.py --task=<task> --num_envs=<envs> --checkpoint=-1` |

For specific commands and detailed training profiles, see the individual task documentation files.

## BUG Report

1. **Issue**: Looks like the less rollout environments it is set, the `step_rollout` is faster (main_env*rollout_envs=Const).

**Problem Location**: For rollout environments, too much robot gather together, causing the following warning:
/buildAgent/work/99bede84aa0a52c2/source/gpubroadphase/src/PxgAABBManager.cpp (1048) : invalid parameter : The application needs to increase PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity to 779948463 , otherwise, the simulation will miss interactions

Detailed information about 0 env spacing can be found in the [issue discussion](https://forums.developer.nvidia.com/t/issue-with-environment-spacing-and-pxgdynamicsmemoryconfig-foundlostaggregatepairscapacity/198272/9).

**TODO**: For rollout envs, try to set slightly different position.

## RSL-RL Training Guides

RSL-RL supports two main training paradigms for legged robots: Actor-Critic with PPO and Teacher-Student Distillation. Here's how to configure each approach in your task configs.

### 1. Actor-Critic with PPO

This is the standard approach for training policies with full observational access during both training and deployment.

**Configuration Classes:**

```python
class AnymalCRoughTeacherCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'      # Standard actor-critic architecture
        algorithm_class_name = 'PPO'           # Proximal Policy Optimization

class AnymalCRoughTeacherCfg(LeggedRobotCfg):
    class env:
        num_observations = 235                 # Actor observation
        num_privileged_obs = None              # Critic observation，if not set, Actor and Critic share the same observation
```

**Key Points:**
- `num_observations`: Total observation dimension available to the policy
- `num_privileged_obs = None`: All observations are treated equally (no asymmetric training)
- Used when the deployed policy has access to all sensor information

### 2. Teacher-Student Distillation

This approach trains a "teacher" policy with privileged information, then distills knowledge to a "student" policy that only uses deployable observations.

**Configuration Classes:**

```python
class AnymalCRoughStudentCfg(AnymalCRoughCfg):
    class env(AnymalCRoughCfg.env):
        num_observations = 144                  # Student obs: proprio only (48 × 3 history)
        num_privileged_obs = 235               # Teacher obs: proprio + height scan (48 + 187)

class AnymalCRoughStudentCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'StudentTeacher'   # Dual-policy architecture
        algorithm_class_name = 'Distillation'  # Knowledge distillation algorithm
```

**Key Points:**
- `num_observations`: Observation space available to the student (deployable sensors only)
- `num_privileged_obs`: Full observation space available to the teacher (including privileged info)
- Teacher uses privileged information (e.g., height maps, ground truth state) during training
- Student learns to mimic teacher behavior using only onboard sensors
- Deployed policy uses only the student network

### Observation Space Breakdown

| Component | Description | Typical Size |
|-----------|-------------|--------------|
| **Proprioceptive** | Joint positions, velocities, IMU data | 48 dims |
| **History Buffer** | Previous timestep observations | × 3 timesteps |
| **Height Scan** | Terrain elevation around robot | 187 dims |
| **Contact Forces** | Ground contact information | Variable |

### Training Workflow

1. **PPO Training**: Train directly with target observation space
2. **Distillation Training**: 
   - First train teacher with privileged observations
   - Then train student to match teacher's actions using limited observations
   - Deploy only the student policy