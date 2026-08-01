# EL4090 Envelope Viewer

这个目录里的 `draw_envelope.py` 用来查看 `el_4090` 从 spider 默认姿态到 mammal 姿态之间的整体空间包络。

默认配置来自：

```text
legged_gym/envs/el_4090/spider_envelop/el4090_spider_config.py
```

URDF 会跟随 env 配置里的 `asset.file`，当前解析到：

```text
resources/robots/el_4090/urdf/el_4090.urdf
```

## 直接查看

在 `legged_gym/legged_gym` 目录下运行：

```bash
python utils/envelop/draw_envelope.py
```

默认显示包络面，并叠加一套 mammal 姿态下的机身和腿部模型。不显示内部采样点，也不记录点云文件。

## 保存图片

```bash
python utils/envelop/draw_envelope.py --output robot_envelope.png
```

## 常用参数

```bash
python utils/envelop/draw_envelope.py --samples 200
python utils/envelop/draw_envelope.py --mode path
python utils/envelop/draw_envelope.py --show-points
python utils/envelop/draw_envelope.py --no-robot
```

- `--samples`: 内部用于构建包络的姿态数量，数值越大越细，但越慢。
- `--mode box`: 默认模式，各关节在两套姿态之间独立取范围。
- `--mode path`: 只沿默认姿态到 mammal 姿态的一条插值路径构建包络。
- `--no-robot`: 只显示包络面，不叠加 mammal 姿态机器人。
- `--max-triangles-per-link`: 控制叠加机器人模型的三角面片数量，数值越大越细，但越慢。
- `--envelope-alpha`: 控制包络面的透明度，默认 `0.14`。
- `--show-points`: 额外显示内部 mesh 顶点点云，调试时用。
- `--live`: 边计算边刷新窗口，默认关闭。

## morphology prior 检查

`morphology_prior.py` 会读取 `spider_envelop/el4090_spider_config.py` 里的 condition 范围，按 env 中相同逻辑随机采样 envelope condition。当前 condition 用 `front_width / middle_width / back_width` 三个半宽度表示左右对称包络，并根据 footprint 大小分别计算前/中/后三段 `morphology_*_prior`。

```bash
python utils/envelop/morphology_prior.py --samples 8 --seed 4090
```

输出包括每个 condition 的采样 min/mean/max、第一组样例值，并默认显示一张 3D 图：包络空间线框 + 当前 morphology prior 对应的机器人状态。同时默认保存第一组样例的 3D 可视化到：

```text
utils/envelop/morphology_prior_3d.png
```

## HAA 摆动范围

`network/haa_swing_range.py` 提供三种相同输入/输出契约的实现。输入为 8 维
envelope + morphology condition，输出为 `[batch, 6, 2]`，最后一维依次为
每条腿 HAA 的下限和上限。

- `AnalyticHaaRangeEstimator`：解析计算，训练环境默认使用。
- `MonteCarloHaaRangeEstimator`：随机采样验证或生成标签。
- `HaaRangeNetwork`：将解析/采样结果蒸馏为小型 MLP。

训练网络：

```bash
python utils/envelop/network/haa_swing_range.py \
  --output logs/haa_range.pt \
  --labels analytic \
  --steps 2000
```

若要在 `el4090_envelop_2` 中使用采样或网络实现，修改
`El4090Envelop2Cfg.haa_swing_range.method` 为 `monte_carlo` 或 `network`；
网络模式还需设置 `network_checkpoint`。

网络测试与可视化：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --output utils/envelop/network/haa_swing_range_visualization.png \
  --save-checkpoint logs/haa_range.pt
```

如果已有模型，使用
`--checkpoint logs/haa_range.pt`，脚本将跳过训练。默认生成 3D 图：显示姿态
先验对应的机器人、包络棱柱、六条腿在网络预测 HAA 上下界之间扫过的半透明
空间，以及每个足端的摆动轨迹。添加 `--plot-2d` 可以查看解析法、Monte
Carlo 和网络上下界的数值对比图。

添加 `--interactive` 会打开交互式 3D UI，可随机包络、切换解析/Monte
Carlo/网络三种方法，并在同一个窗口中旋转、缩放和刷新当前结果。
