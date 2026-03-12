# EL_4090 模块说明

简要说明 EL_4090 目录下主要环境/实现的架构：`safe`、`spider_normal`、`thirdparty`。

## Overview

###  
  EL_4090 是基于 legged_gym 的六足机器人（EL_4090/Spider）实现变种集合，包含带安全层的训练环境、常规模型实现与第三方接口。
- **目标读者**: 开发者与算法工程师，需理解各模块如何组织观测、动作、控制与安全层。

**Safe 架构**

- **用途**: 在策略前/后插入 ATACOM 的安全层，用以保证约束满足（关节限位、速度、扭矩、基座姿态等）。
- **关键文件**:
  - `el_4090_safe_config.py`: 环境配置（观测维度、是否使用高度测量、ATACOM slack 大小等）。
  - `el_4090_safe_symmetry.py`: 对称数据增强（左右/前后镜像）并对 `u_mu`（ATACOM slack）按约束类型做正确交换/取反。
  - 安全层实现（通常在 `utils/atacom.py` 或对应模块）负责从策略动作产生受限动作与 slack 向量。
- **数据流**:
  1. 策略输出原始动作 `a`。
  2. ATACOM 安全层计算约束 slack `u_mu` 与投影/修正后的动作 `a_safe`。
  3. 环境接收 `a_safe`，并将 `u_mu` 附在观测末尾（用于监督/诊断或 critic）。
- **观测布局**: 当 `measure_heights=False` 时，默认核心观测 66 维（基座速度、角速度、重力投影、命令、18 DOF 的位置/速度/上次动作）后接 `u_mu`（77 维）。若 `measure_heights=True`，高度测量插入在 core 与 `u_mu` 之间。
- **注意点**: 对称增强必须对 `u_mu` 做约束感知的变换（不同约束段需要 swap/negate/保持），以保证镜像观测与物理意义一致。

**Spider_Normal 架构**

- **用途**: `spider_normal`（或称 `spider_nominal` / `spider_normal` 实现）是没有额外 ATACOM 风险层的常规模型，用于快速训练和基线比对。
- **关键文件**:
  - `el4090_spider_config.py`（或类似命名）: 基本环境参数（obs/action 大小、控制参数、decimation 等）。
  - `elspider.py`（或环境实现）: 环境核心步进逻辑、状态-观测映射、奖励与终止条件。
  - `scripts/train.py`: 训练入口，使用对应的 `PPO` / 算法配置加载 `spider_normal` 环境。
- **数据流**:
  1. 策略输出动作 `a`（通常为目标位置或关节目标）。
  2. 环境将 `a` 应用到低级 PD 控制器（或 actuator network），得到实际施加的 torque/position。
  3. 环境返回下一步观测（不包含 `u_mu`）。
- **观测布局**: 仅包含核心 66 维（除非配置开启高度测量），无 ATACOM slack 附带。

**ThirdParty（第三方）集成**

- **用途**: 放置第三方实现、外部控制器或外部仿真/工具的适配层（例如外部动力学模型、actuator 网络、或供应商提供的控制包）。
- **典型内容**:
  - `resources/actuator_nets/`：预训练的 actuator 网络模型（如 anydrive_v3_lstm.pt）。
  - 第三方驱动或接口脚本：封装加载、推断与对接代码（将第三方输出适配为环境动作）。
- **集成要点**:
  - 保持统一的动作/观测接口：任何第三方模块输出必须被封装成合法的 `action` 向量（18 维，按腿/关节顺序）。
  - 时序与 decimation：第三方控制器需遵守环境 `decimation`（policy 更新与模拟步之间的关系）。
  - 维护 deterministic 行为：对比测试时，记录第三方模块版本与超参以保证可复现性。
