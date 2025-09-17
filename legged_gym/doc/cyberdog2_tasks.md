# CyberDog2 Robot Tasks

## Environment Setup
```bash
conda activate diffuseloco
```

## CyberDog2 Stand

Train CyberDog2 for standing behavior.

```bash
python legged_gym/scripts/train.py --task=cyber2_stand --num_envs=6144 --resume --headless
```