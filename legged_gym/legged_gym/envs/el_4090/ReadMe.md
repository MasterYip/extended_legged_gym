# EL_4090 环境说明

`envs/el_4090` 放置 EL_4090 六足机器人在不同形态/约束设定下的训练环境。当前主要变体包括：

- `spider_nomal`: 基础 spider 形态环境，用作常规基线。
- `spider_mammal`: 偏 mammal 形态的环境变体。
- `spider_both`: 同时覆盖 spider/mammal 形态的环境变体。
- `spider_envelop`: 新增的 envelope 条件环境，用可采样的足端活动范围约束策略。
- `safe`: 带 ATACOM 安全层的环境。
- `thirdparty`: 第三方对称增强/接口相关代码。

本文重点说明 `spider_envelop`，因为它在原有速度命令之外新增了 envelope condition，并用这些 condition 同时驱动观测、形态先验、奖励约束和可视化。

## Spider Envelop 目标

`spider_envelop` 的核心目标是让策略学会在不同足端活动范围下运动。环境每隔一段时间采样一组 envelope condition，表示机器人身体坐标系下允许足端落点出现的平面范围。

策略会在 observation 中直接看到这些 condition，因此它不是只学一个固定形态，而是学一个条件策略：

```text
policy(obs, velocity command, envelope condition) -> action
```

如果足端超出当前 envelope，环境通过 `envelope_constraint` 奖励项给惩罚。这样可以训练策略根据不同结构范围自动调整步态和腿部摆动。

## Command 和 Condition 布局

`spider_envelop` 中 `commands.num_commands = 4 + 8`，总共 12 维：

```text
commands[0]  lin_vel_x
commands[1]  lin_vel_y
commands[2]  ang_vel_yaw
commands[3]  heading
commands[4:12] envelope condition
```

其中 condition 的顺序由 `condition_names` 定义：

```python
condition_names = [
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
    "morphology_front_prior",
    "morphology_middle_prior",
    "morphology_back_prior",
]
```

前 5 个是 envelope 的几何边界，后 3 个是由 envelope 推导出的形态先验。

## Envelope 几何定义

Envelope 是机器人 base yaw 坐标系下的 2D 足端范围。代码中会把 condition 转成 6 个边界点：

```text
(forward_limit,  front_width)
(0,              middle_width)
(backward_limit, back_width)
(backward_limit, -back_width)
(0,              -middle_width)
(forward_limit,  -front_width)
```

也就是说：

- `front_width`: 身体前方区域的左右半宽。
- `middle_width`: 身体中部区域的左右半宽。
- `back_width`: 身体后方区域的左右半宽。
- `forward_limit`: 足端允许到达的最前方 x 边界。
- `backward_limit`: 足端允许到达的最后方 x 边界，通常为负值。

当前默认范围在 `spider_envelop/el4090_spider_config.py` 中：

```python
front_width = [0.3, 0.6]
middle_width = [0.3, 0.7]
back_width = [0.3, 0.6]
forward_limit = [0.6, 0.9]
backward_limit = [-0.9, -0.6]
```

环境在计算超界惩罚时，会把足端位置转换到 base yaw 坐标系下，再检查足端是否落在这个由前/中/后三段线性插值得到的范围内。

## Morphology Prior 设计

`morphology_front_prior`、`morphology_middle_prior`、`morphology_back_prior` 不直接随机采样，而是由前 5 个 envelope 几何参数计算得到。

它们的取值范围是 `[0, 1]`：

- `0` 更偏 spider 默认姿态。
- `1` 更偏 mammal 默认姿态。

默认模式是：

```python
morphology_prior_mode = "directional_ratio"
```

这个模式会把横向宽度和前后 reach 的关系转成形态先验。直觉上：

- 横向范围越宽，越偏 spider。
- 前后 reach 越强，越偏 mammal。
- middle 部分主要由 `middle_width` 决定。

这些 prior 会用于两个地方：

1. **默认关节目标插值**

   `embedded_state_default_dof_pos` 会根据 prior 在 spider 默认关节角和 mammal 默认关节角之间插值：

   ```text
   target = spider_default + prior * (mammal_default - spider_default)
   ```

   front/middle/back 三组腿分别使用对应的 prior。

2. **机身高度目标插值**

   base height target 会在 spider 高度和 mammal 高度之间插值：

   ```python
   base_height_spider_target = 0.53
   base_height_mammal_target = 0.64
   ```

这样 envelope 不只是一个惩罚边界，也会影响机器人应该采取的身体形态。

## Observation 设计

`spider_envelop` 的基础观测维度为 74，开启 LiDAR 后总维度为：

```python
num_observations = 74 + 11 * 17
```

基础观测由以下部分拼接：

```text
base_lin_vel                 3
base_ang_vel                 3
projected_gravity            3
lin_vel_x command            1
lin_vel_y command            1
ang_vel_yaw command          1
envelope condition           8
dof_pos - condition target   18
dof_vel                      18
last actions                 18
```

注意：condition 在 observation 中直接使用 `commands[:, condition_start_idx:condition_end_idx]` 输入。因为 condition 在采样时已经按配置范围生成，所以这里不再额外裁剪。

噪声配置中 commands 段被置为 0 噪声，包含速度命令和 envelope condition，避免策略看到被扰动的目标条件。

## Condition 采样流程

正常训练时，`_resample_commands(env_ids)` 会周期性重新采样速度命令和 envelope condition。

Condition 采样步骤：

1. 在 `[0, 1]` 中随机采样 8 维向量。
2. 映射到 `condition_low ~ condition_high`。
3. 用 `_set_morphology_prior_from_envelope()` 根据前 5 个几何参数重算后 3 个 morphology prior。
4. 写入 `commands[:, 4:12]`。
5. 根据新 condition 更新 `embedded_state_default_dof_pos`。

因此，配置里的 `morphology_*_prior` 范围用于声明合法范围，但正常情况下它们会被 envelope 几何覆盖，而不是独立随机决定。

## Envelope Reward

新增奖励项：

```python
envelope_constraint = -10.0
```

对应实现是 `_reward_envelope_constraint()`。它会：

1. 取当前 condition 并生成 envelope 边界。
2. 将每个足端位置转换到 base yaw 坐标系。
3. 根据足端 x 所处的前半区/后半区，线性插值得到当前 x 下的左右边界。
4. 对超出 x/y 边界的距离平方求平均。
5. 只有移动命令超过 `envelope_constraint_min_command` 时才启用惩罚。

相关配置：

```python
envelope_constraint_margin = 0.0
envelope_constraint_min_command = 0.15
```

`margin` 可以扩大或收紧判定边界；`min_command` 用来避免站立时也强行惩罚足端范围。

## Envelope 可视化

可视化开关在 commands 配置中：

```python
envelope_debug_viz = True
envelope_debug_env_ids = [0]
envelope_debug_ground_z_offset = 0.02
envelope_debug_color = (0.0, 0.85, 1.0)
envelope_debug_line_radius = 0.012
envelope_debug_line_samples = 8
```

当前可视化会把 envelope 画成贴近地面的 2D 轮廓线：

- 跟随机器人 `x/y` 位置移动。
- 跟随机器人 yaw 旋转。
- 不再画高度柱体，只画地面 footprint。
- `ground_z_offset` 默认 2 cm，用来避免线和地面重叠闪烁。

这个视图主要用于观察足端是否越过当前 condition 对应的 envelope 范围。

## Morphology Reachability Test

`morphology_reachability_test` 是 debug/test 功能，不是正常训练采样逻辑。

```python
morphology_reachability_test = False
morphology_reachability_test_mode = "corners"
morphology_reachability_resample_steps = 600
morphology_reachability_print_interval = 100
```

打开后，环境会按固定模式采样 condition，并打印指定 env 的可达性状态。`morphology_reachability_test_mode` 有三种：

- `"center"`: 所有 condition 取范围中点。
- `"random"`: 每次随机采样 condition。
- `"corners"`: 对非 `morphology_` 的几何 condition 取 low/high 组合，用来测试参数空间角点。

`corners` 适合检查最窄、最宽、最靠前、最靠后的极端 envelope 下，默认形态目标和足端范围是否合理。

## 与其他变体的关系

`spider_envelop` 继承自 `ElSpider`，不是 ATACOM safe 环境。它的约束来自 reward 和 condition，而不是动作投影安全层。

和普通 spider 环境相比，它主要新增：

- command 从原来的速度/heading 扩展为速度 + envelope condition。
- observation 增加 8 维 condition。
- DOF position observation 使用 `dof_pos - embedded_state_default_dof_pos`，其中默认姿态由 condition 决定。
- reward 增加 envelope footprint 超界惩罚。
- debug viewer 增加跟随机器人移动的地面 envelope 轮廓。
- 提供 morphology reachability test 用于检查 condition 设计是否可达。

`safe` 环境仍然用于 ATACOM 安全层实验；`spider_envelop` 更适合研究“给策略一个结构/包络条件，让策略在该条件下学习运动”。
