# Franka Tasks

## Franka

```bash
python legged_gym/scripts/train.py --task=franka --num_envs=40 --resume --headless
```

## Franka Batch Rollout

```bash
python legged_gym/scripts/train.py --task=franka_batch_rollout --num_envs=40 --resume --headless
python legged_gym/scripts/play.py --task=franka_batch_rollout --num_envs=48 --checkpoint=-1
```