# EL4090 HAA Swing Range

`haa_swing_range.py` 用于根据包络信息与姿态先验，为 EL4090 六条腿生成各自的
HAA 合法摆动区间：

```text
[HAA lower, HAA upper]
```

文件提供三种实现：

1. 解析几何法 `AnalyticHaaRangeEstimator`
2. Monte Carlo 随机采样法 `MonteCarloHaaRangeEstimator`
3. 神经网络回归法 `HaaRangeNetwork`

`spider_envelop_2` 默认加载已训练网络，也可以在配置中切换为解析法或 Monte
Carlo。

## 1. 输入与输出

HAA 范围网络的输入是 8 维 envelope + morphology condition：

```text
[
    front_width,
    middle_width,
    back_width,
    forward_limit,
    backward_limit,
    morphology_front_prior,
    morphology_middle_prior,
    morphology_back_prior,
]
```

批量输入张量形状：

```text
[batch_size, 8]
```

输出张量形状：

```text
[batch_size, 6, 2]
```

六条腿默认顺序：

```text
RF, RM, RB, LF, LM, LB
```

输出最后一维含义：

```text
output[..., 0] = HAA lower
output[..., 1] = HAA upper
```

这里的 8 维输入只进入独立的 HAA 范围网络，不再进入 locomotion policy 的
observation，也不再存放在 `env.commands` 中。

`spider_envelop_2` 的策略观测已恢复为 66 维：

```text
base linear velocity:   3
base angular velocity:  3
projected gravity:      3
command [vx, vy, yaw]:  3
DOF position error:    18
DOF velocity:          18
previous actions:      18
-------------------------
total:                 66
```

包络数据由下面的独立状态类持有：

```text
envs/el_4090/spider_envelop_2/envelope_condition.py
EnvelopeConditionState
```

数据流为：

```text
EnvelopeConditionState: 8-D envelope/prior
        -> HaaRangeNetwork
        -> six [lower, upper] ranges
        -> HAA reward constraints

env.commands: [vx, vy, yaw_rate]
        -> 66-D locomotion policy observation
```

外部模块以后可以通过环境接口写入包络：

```python
env.set_envelope_condition(values, env_ids, derive_priors=True)
```

其中 `derive_priors=True` 会根据前五个包络量重新推导三个 morphology prior。

例如：

```text
RF: [-1.20, 0.10]
RM: [ 0.05, 1.25]
RB: [ 0.20, 1.30]
LF: [-1.18, 0.08]
LM: [ 0.04, 1.24]
LB: [ 0.19, 1.29]
```

## 2. `HaaRangeConfig`

`HaaRangeConfig` 保存三种估计器共同使用的几何参数和关节限制。

主要参数：

```python
joint_lower = -3.0
joint_upper = 3.0

leg_reach = 0.55

front_hip_offset = 0.10
middle_hip_offset = 0.20
back_hip_offset = 0.10

spider_swing_limit = 1.05
mammal_swing_limit = 0.45
minimum_half_range = 0.05
```

蜘蛛形态的 HAA 默认位置：

```python
spider_haa = (
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
)
```

哺乳形态的 HAA 默认位置：

```python
mammal_haa = (
    -1.308, 1.308, 1.308,
    -1.308, 1.308, 1.308,
)
```

配置初始化时会检查：

- condition 是否包含三个包络宽度。
- condition 是否包含前、中、后三个 morphology prior。
- HAA 物理下限是否小于物理上限。
- 腿长是否为正数。
- 腿名称、蜘蛛 HAA 和哺乳 HAA 的数量是否一致。

## 3. 解析几何法

解析法由 `AnalyticHaaRangeEstimator` 实现，是训练环境的默认方法。

### 3.1 根据姿态先验确定 HAA 中心

每条腿的 HAA 中心在蜘蛛姿态和哺乳姿态之间插值：

```text
center =
    spider_HAA
    + morphology_prior * (mammal_HAA - spider_HAA)
```

当 morphology prior 为 0：

```text
center = spider_HAA
```

当 morphology prior 为 1：

```text
center = mammal_HAA
```

当 morphology prior 为 0.5：

```text
center = 0.5 * spider_HAA + 0.5 * mammal_HAA
```

不同位置的腿使用不同的 prior：

```text
RF、LF -> morphology_front_prior
RM、LM -> morphology_middle_prior
RB、LB -> morphology_back_prior
```

### 3.2 根据包络宽度计算几何限制

解析法使用一个简化的腿部横向几何关系：

```text
hip_offset + leg_reach * sin(abs(HAA - center))
    <= envelope_width
```

整理后得到 HAA 相对中心的最大角度偏移：

```text
geometric_limit =
    asin(
        clamp(
            (envelope_width - hip_offset) / leg_reach,
            0,
            1
        )
    )
```

不同位置的腿使用对应的包络宽度：

```text
RF、LF -> front_width
RM、LM -> middle_width
RB、LB -> back_width
```

因此：

- 包络越宽，允许的 HAA 摆动范围越大。
- 包络越窄，允许的 HAA 摆动范围越小。
- 中腿的髋关节横向偏置更大，相同宽度下可能得到更小的角度范围。

### 3.3 morphology 对摆动范围的限制

蜘蛛形态允许较大的 HAA 摆动：

```text
spider_swing_limit = 1.05 rad
```

哺乳形态允许较小的 HAA 摆动：

```text
mammal_swing_limit = 0.45 rad
```

当前形态的限制通过 morphology prior 插值得到：

```text
morphology_limit =
    spider_swing_limit
    + morphology_prior
      * (mammal_swing_limit - spider_swing_limit)
```

所以 morphology prior 越接近 1，HAA 中心越接近哺乳姿态，允许的摆动半范围也
越小。

### 3.4 与物理关节限位求交集

中心位置距离最近物理关节限位的距离为：

```text
joint_limit = min(
    center - joint_lower,
    joint_upper - center
)
```

最终半范围取三个限制的交集：

```text
half_range = min(
    geometric_limit,
    morphology_limit,
    joint_limit
)
```

最终上下界：

```text
lower = center - half_range
upper = center + half_range
```

所以解析结果同时满足：

- 横向包络宽度限制。
- morphology 姿态先验限制。
- HAA 物理关节限位。

## 4. Monte Carlo 方法

Monte Carlo 方法由 `MonteCarloHaaRangeEstimator` 实现。

它不是完整的 Isaac Gym 仿真或 URDF 碰撞采样，而是在 HAA 关节空间中随机
采样，用于验证解析结果或生成带采样误差的训练标签。

计算步骤：

1. 使用相同几何模型得到 HAA 中心和允许半范围。
2. 在物理关节范围内随机生成大量 HAA：

   ```text
   q ~ Uniform(joint_lower, joint_upper)
   ```

3. 判断每个样本是否满足：

   ```text
   abs(q - center) <= half_range
   ```

4. 对可行样本排序。
5. 使用可行样本的最小值和最大值作为估计范围。
6. 如果没有采到可行样本，则回退到解析结果。

默认采样数量：

```text
2048 samples / condition / leg
```

还可以配置 `quantile`，忽略采样结果两侧的少量极端点。

Monte Carlo 适合：

- 检查解析上下界。
- 生成随机采样标签。
- 分析采样数量和范围误差。
- 对神经网络进行采样教师蒸馏。

它不适合在数千个并行环境的每个仿真步中执行，因此环境默认不使用 Monte
Carlo。

## 5. 神经网络方法

神经网络由 `HaaRangeNetwork` 实现。

它不是传统意义上的特征 encoder，而是一个带物理输出约束的 HAA 范围回归
网络。

默认结构：

```text
8-dimensional condition
    -> Linear(8, 128)
    -> ELU
    -> Linear(128, 128)
    -> ELU
    -> Linear(128, 12)
```

12 个原始输出表示：

```text
6 legs * [raw_center, raw_half_range]
```

### 5.1 受约束的中心输出

中心通过 sigmoid 映射到物理关节范围：

```text
center =
    joint_lower
    + sigmoid(raw_center)
      * (joint_upper - joint_lower)
```

因此：

```text
joint_lower <= center <= joint_upper
```

### 5.2 受约束的半范围输出

先计算中心距离最近物理限位的距离：

```text
max_half_range = min(
    center - joint_lower,
    joint_upper - center
)
```

再计算网络输出的半范围：

```text
half_range =
    sigmoid(raw_half_range) * max_half_range
```

最后生成：

```text
lower = center - half_range
upper = center + half_range
```

这种受约束输出从网络结构上保证：

```text
joint_lower <= lower <= upper <= joint_upper
```

即使网络尚未充分训练，也不会出现：

- HAA 下界大于上界。
- HAA 范围超出物理关节限位。
- 负的摆动半范围。

## 6. 网络训练

`fit_estimator()` 使用教师蒸馏方式训练网络。

训练程序不会维护一套独立的硬编码包络范围，而是通过 AST 直接读取：

```text
envs/el_4090/spider_envelop/el4090_spider_config.py
```

读取内容包括：

- `commands.condition_names`
- `commands.ranges` 中全部 condition 上下界
- `morphology_prior_mode`
- `morphology_prior_weights`
- `morphology_middle_front_follow_weight`

因此 `spider_envelop` 中的字段顺序或包络范围修改后，重新运行训练程序会自动
使用新配置。

当前读取到的范围是：

```text
front_width:             [ 0.3,  0.6]
middle_width:            [ 0.3,  0.7]
back_width:              [ 0.3,  0.6]
forward_limit:           [ 0.6,  0.9]
backward_limit:          [-0.9, -0.6]
morphology_front_prior:  [ 0.0,  1.0]
morphology_middle_prior: [ 0.0,  1.0]
morphology_back_prior:   [ 0.0,  1.0]
```

首先在这些 condition 范围内随机采样：

```text
condition =
    condition_low
    + random * (condition_high - condition_low)
```

随后使用与环境 `_set_morphology_prior_from_envelope()` 相同的公式，根据前五个
实际包络参数重新计算三个 morphology prior。三个 prior 不是独立随机训练
变量，这样可以避免产生环境中不存在的 envelope-prior 组合。

然后由教师估计器生成标签：

```python
target = estimator(condition)
```

教师可以选择：

```text
AnalyticHaaRangeEstimator
```

或者：

```text
MonteCarloHaaRangeEstimator
```

网络预测：

```python
prediction = network(condition)
```

训练损失是六条腿上下界的均方误差：

```text
loss = mean((prediction - target) ** 2)
```

默认优化器和学习率：

```text
optimizer = Adam
learning_rate = 3e-4
```

默认训练参数：

```text
steps = 2000
batch_size = 1024
```

### 训练命令

使用解析教师：

```bash
python utils/envelop/network/haa_swing_range.py \
  --output utils/envelop/logs/haa_range.pt \
  --labels analytic \
  --steps 2000 \
  --batch-size 1024
```

使用 Monte Carlo 教师：

```bash
python utils/envelop/network/haa_swing_range.py \
  --output utils/envelop/ogs/haa_range_mc.pt \
  --labels monte_carlo \
  --samples 4096 \
  --steps 2000
```

## 7. Checkpoint

保存网络：

```python
network.save_checkpoint("haa_range.pt")
```

checkpoint 包含：

- 网络权重 `state_dict`。
- 完整 `HaaRangeConfig`。
- 隐藏层尺寸 `hidden_dims`。

加载网络：

```python
network = HaaRangeNetwork.from_checkpoint(
    "haa_range.pt",
    device="cuda",
)
```

环境加载 checkpoint 时还会检查：

- condition 输入顺序是否与环境一致。
- 六条腿输出顺序是否与环境一致。

避免网络输出与环境腿序发生错位。

## 8. 包络变化如何影响 HAA 范围

当前实现中，三个横向包络宽度直接影响 HAA 范围：

```text
front_width  -> RF、LF 的 HAA 半范围
middle_width -> RM、LM 的 HAA 半范围
back_width   -> RB、LB 的 HAA 半范围
```

三个 morphology prior 同时影响：

- HAA 摆动中心。
- morphology 对应的最大摆动半范围。

`forward_limit` 和 `backward_limit` 不直接进入当前 HAA 解析几何公式，但在
环境中会参与 morphology prior 的计算。因此正常环境中的依赖关系是：

```text
forward_limit / backward_limit 改变
    -> morphology prior 重新计算
    -> HAA 中心和摆动半范围改变
```

如果只修改 `forward_limit` 或 `backward_limit`，同时人为保持三个
morphology prior 不变，那么当前解析教师不会直接改变 HAA 范围。

这是因为当前简化模型认为：

- HAA 主要控制横向摆动。
- 横向包络宽度直接约束 HAA。
- 纵向范围主要由 HFE 和 KFE 负责。

当前网络虽然接收完整的 8 维 condition，但其解析教师主要通过宽度和
morphology prior 生成 HAA 标签。它没有执行完整 URDF 逆运动学、碰撞检测
或足端可达空间优化。

## 9. 3D 可视化

`visualize_haa_swing_range.py` 会显示：

- morphology prior 对应的机器人姿态。
- 当前 envelope 的 3D 棱柱。
- 六条腿在网络预测 HAA 上下界之间扫过的半透明空间。
- 每条腿的 HAA 上下界姿态。
- 六个足端的摆动轨迹。
- 每条腿预测的 `[lower, upper]`。

使用已有 checkpoint：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --checkpoint logs/haa_range.pt
```

默认输出：

```text
utils/envelop/network/haa_swing_range_visualization.png
```

没有 checkpoint 时，程序会先训练网络，再生成 3D 图：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --save-checkpoint logs/haa_range.pt
```

查看解析法、Monte Carlo 和网络的二维数值对比：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --checkpoint logs/haa_range.pt \
  --plot-2d \
  --output utils/envelop/network/haa_swing_range_diagnostic.png
```

### 交互式 3D UI

启动交互界面：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --interactive \
  --checkpoint logs/haa_range.pt
```

如果不提供 checkpoint，程序会先训练一个网络，然后打开 UI：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --interactive \
  --train-steps 2000
```

界面功能：

- `Random Envelope`：按照 `spider_envelop` 的配置范围随机生成新的包络，并用
  环境公式重新推导三个 morphology prior。
- `Analytic`：使用解析几何法生成六条腿的 HAA 范围。
- `Monte Carlo`：使用随机关节采样生成 HAA 范围。
- `Network`：使用训练后的神经网络生成 HAA 范围。
- `Save Snapshot`：将当前 UI 画面保存为
  `utils/envelop/network/haa_swing_range_ui_snapshot.png`。
- 在 3D 区域按住鼠标拖动：旋转观察方向。
- 鼠标滚轮：放大或缩小。

随机包络或切换方法时不会创建新窗口，而是在同一个 3D 坐标轴中刷新：

- 当前机器人姿态。
- 当前 3D envelope。
- 六条腿的 HAA 扫掠曲面。
- 六个足端的摆动轨迹。
- 每条腿的 HAA 上下界标签。
- 左侧当前 8 维 condition 数值。

可以指定 UI 初始方法：

```bash
python utils/envelop/network/visualize_haa_swing_range.py \
  --interactive \
  --checkpoint logs/haa_range.pt \
  --method monte_carlo
```

## 10. 方法的定位与局限

当前方法是一个适合大规模并行训练的高效简化模型。

优点：

- 解析法完全向量化。
- 可以在数千个环境中批量计算。
- 明确考虑横向包络、姿态先验和物理限位。
- 网络输出天然满足上下界顺序和物理限位。
- 支持解析、随机采样和网络三种互相验证的方法。

局限：

- Monte Carlo 只在 HAA 关节空间采样，不是完整机器人仿真。
- 没有在范围生成阶段执行自碰撞检查。
- 没有直接计算足端与完整 3D 包络的交集。
- 没有联合优化 HAA、HFE 和 KFE。
- `forward_limit` 和 `backward_limit` 主要通过 morphology prior 间接影响
  HAA。

训练环境中的足端包络奖励仍会独立检查实际足端是否超出 envelope，因此
HAA 范围约束与足端包络约束是两层互补约束。
