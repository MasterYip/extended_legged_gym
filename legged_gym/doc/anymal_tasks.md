# Anymal Robot Tasks

## Environment Setup
```bash
conda activate diffuseloco
```

## Teacher-Student Pipeline AnymalC

Train AnymalC using teacher-student distillation for improved generalization with limited observations.

### Teacher Model Training

Train the teacher model with full privileged observations (235 dims including terrain height scans).

**Training Commands:**
```bash
python legged_gym/scripts/train.py --task=anymal_c_rough_teacher --num_envs=4096 --resume --headless
python legged_gym/scripts/play.py --task=anymal_c_rough_teacher --num_envs=48 --checkpoint=-1
```

**Training Profile:**
- Uses full privileged observations including terrain height measurements
- 235 observation dimensions (48 proprioceptive + 187 height scan)
- Training epochs: ~550 for optimal performance

### Student Model Training

Train the student model using distillation from the pre-trained teacher model.

**Prerequisites:**
- Trained teacher model checkpoint (update path in student config)
- Teacher model path: `/home/user/CodeSpace/Python/PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/logs/rough_anymal_c_teacher/Jun06_11-09-08_/model_550.pt`

**Training Commands:**
```bash
python legged_gym/scripts/train.py --task=anymal_c_rough_student --headless --resume
python legged_gym/scripts/play.py --task=anymal_c_rough_student --num_envs=48 --checkpoint=-1
```

**Training Profile:**
- Student observations: 144 dims (48 proprioceptive × 3 history steps)
- Uses distillation algorithm instead of PPO
- Max iterations: 1500
- Learning rate: 1e-3
- Loss type: MSE between teacher and student policies

**Key Features:**
- **Teacher**: Full terrain perception with height scans
- **Student**: History-based proprioceptive observations only
- **Distillation**: Knowledge transfer from teacher to student
- **Deployment**: Student model can run without terrain sensors

## BatchRolloutAnymal

Train Anymal with batch rollout capability for trajectory optimization.

**Training Commands:**
```bash
python legged_gym/scripts/train.py --task=anymal_c_batch_rollout --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=anymal_c_batch_rollout --num_envs=32 --checkpoint=-1
```

**Flat Terrain Variant:**
```bash
python legged_gym/scripts/train.py --task=anymal_c_batch_rollout_flat --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=anymal_c_batch_rollout_flat --num_envs=32 --checkpoint=-1
```

## BasePoseAdapt AnymalC

Train Anymal C with base pose adaptation for collision avoidance.

**Training Commands:**
```bash
python legged_gym/scripts/train.py --task=anymal_c_base_pose_adapt --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=anymal_c_base_pose_adapt --num_envs=48 --checkpoint=-1
```

**Test Base Pose PD Control:**
```bash
python legged_gym/scripts/train.py --task=anymal_c_base_pose_ctrl --num_envs=48
```

## AnymalC Flat Terrain

**Training Epoch:** ~200

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=anymal_c_flat --num_envs=48 --checkpoint=-1
```

## LoadAdapt AnymalC Flat

Train Anymal C with load adaptation on flat terrain.

```bash
python legged_gym/scripts/train.py --task=load_adapt_anymal_c_flat --num_envs=4096 --resume --headless
python legged_gym/scripts/play.py --task=load_adapt_anymal_c_flat --num_envs=1 --checkpoint=-1
```

## Pose AnymalC Flat

Train Anymal C for pose control on flat terrain.

```bash
python legged_gym/scripts/train.py --task=pose_anymal_c_flat --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=pose_anymal_c_flat --num_envs=48 --checkpoint=-1
```

## Stand AnymalC Flat

Train Anymal C for standing behavior on flat terrain.

```bash
python legged_gym/scripts/train.py --task=stand_anymal_c_flat --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=stand_anymal_c_flat --num_envs=48 --checkpoint=-1
```

## AnymalC Rough Terrain

**Training Epoch:** ~200

**Training Profile:**
- 100 epoch: velocity tracking reward grows up

```bash
python legged_gym/scripts/train.py --task=anymal_c_rough --num_envs=6144 --resume --headless
python legged_gym/scripts/play.py --task=anymal_c_rough --num_envs=48 --checkpoint=-1
```

## AnymalC Trajectory Gradient Sampling

Train Anymal C with gradient sampling for trajectory optimization.

```bash
python legged_gym/scripts/train.py --task=anymal_c_traj_grad_sampling --num_envs=6144 --headless --resume 
python legged_gym/scripts/play.py --task=anymal_c_traj_grad_sampling --num_envs=32 --checkpoint=-1
```