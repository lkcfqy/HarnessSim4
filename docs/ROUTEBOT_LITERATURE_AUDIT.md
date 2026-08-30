# RouteBot 相关工作与新颖性审计

检索日期：2026-08-26。本文档只记录已核验的一手论文/官方代码入口，用于约束论文主张，
不以搜索摘要代替最终逐篇全文阅读。

## 直接竞争工作

| 工作 | 已公开贡献 | 与 RouteBot 的重叠 | 对本项目的约束 |
|---|---|---|---|
| [WireCraft (2026)](https://arxiv.org/abs/2606.18097) | 工业 DLO 仿真基准；插接、卡扣布线、槽内铺设；双物理模型；仿真和真实示教；RL/IL/VLA | **高度重叠**：工业线束、clip routing、策略基准 | 不能再把“首个工业 DLO/线束基准”当贡献；必须突出显式语义拓扑、分支身份、反事实测试及跨后端验证，并争取在 WireCraft 上外部复核 |
| [DLO-Lab (ICML 2026)](https://arxiv.org/abs/2606.04206) / [官方代码](https://github.com/UMass-Embodied-AGI/DLO-Lab) | 可微 DLO 物理、材料多样性、任务基准、抓点与长时程分解 | 部分重叠：DLO 基准、拓扑复杂性、抓点敏感性 | HarnessSim4 不能主张物理精度或任务覆盖领先；论文重点必须是工艺语义关系表示和可证伪干预 |
| [Harnessing with Twisting (2024)](https://arxiv.org/abs/2410.10729) | 单臂扭转张紧、Koopman MPC、卡扣插入、固定点切换、真实工业级实验 | 直接重叠：单臂线束入扣 | RouteBot 当前没有力/张紧控制，不能声称优于工业控制系统；应把它列为真实物理强基线/未来硬件对照 |
| [Contact-aware Shaping and Maintenance (2023)](https://arxiv.org/abs/2307.10153) | 双臂、环境接触、形状规划、在线视觉/力状态估计、真实卡扣固定 | 重叠：fixture-aware routing | 必须承认 HarnessSim4 目前缺少真实触觉与在线形状估计；仿真论文主张应限于语义决策层 |
| [Multi-Robot Assembly of DLOs (2025)](https://arxiv.org/abs/2506.22034) | 视觉+触觉、抓取—交接—装夹完整工业流程、真实多机器人 | 重叠：完整 DLO 装配链与多机器人 | “四机器人流水线”本身不是充分新颖性；需要统一表示、迁移收益和反事实稳健性证据 |
| [Multi-Stage Cable Routing (T-RO 2024)](https://sites.google.com/view/cablerouting/home) | 真实 Franka 多视角示教、低层入扣、高层原语选择、失败恢复与公开数据 | 直接重叠：单/多卡扣长时程布线和模仿学习 | 已把固定版本的 1,647 条真实轨迹纳入无泄漏离线审计；因其没有本项目语义干预和统一闭环标签，只能验证数据接口与时序信号 |

## 方法与标准学习基线

- [ACT / ALOHA (RSS 2023)](https://arxiv.org/abs/2304.13705)：CVAE Transformer 预测动作块；
  官方实现见 <https://github.com/tonyzhaozh/act>。本仓库的低维实现只能称为
  `ACT-style state baseline`，差异记录在 `methods/act_chunk/README.md`。
- [Diffusion Policy (RSS 2023 / IJRR 2024)](https://diffusion-policy.cs.columbia.edu/)：条件动作
  扩散、滚动时域与时序建模。若 RouteBot 的动作存在多模态，应作为第二强学习基线。
- [Offline-Online Cable GNN (2022)](https://arxiv.org/abs/2203.15004)：仿真 GNN 动力学模型、在线
  线性残差与 MPC，说明“GNN 用于线缆”本身也不是新颖性。
- [DEFORM](https://openreview.net/forum?id=V5x0m6XDSV)：可微离散弹性杆与学习结合，用于实时
  DLO 建模、跟踪、规划和控制，可作为模型精度与 sim-to-real 讨论依据。

## RouteBot 可保留的论文主张

在上述竞争格局下，最有机会成立的主张不是“又一个线束仿真器”，而是：

> 在物理状态完全相同但夹具绑定到相邻可行线段的精确关系发生变化时，显式关系图策略能够
> 保持正确装配顺序，
> 而只看坐标或普通动作序列的策略更容易产生永久性的错段入扣；这种优势可通过严格配对的
> 反事实干预和跨物理后端复核量化。

这要求最终论文至少完成：

1. 同物理 seed 下九种经独立 Teacher 语义完成审计后冻结的顺序保持线段—卡扣绑定干预；
2. TopoHarness、同参数 Geometry、ACT、Diffusion Policy、Teacher/Random 的公平比较；
3. 100 seed/难度、McNemar 精确检验、效应量和失败类型；
4. MuJoCo 直接闭环，以及优先在 WireCraft 或 DLO-Lab 上做外部任务适配；
5. 清楚区分工艺语义决策贡献与接触力控、材料建模贡献。

## 当前新颖性结论

RouteBot 的“工业线束 clip-routing benchmark”定位已被 WireCraft 明显占先，不能按旧标题投稿。
可行的新标题方向是：

> **Explicit Process Relations Prevent Semantic Misrouting in Long-Horizon Robotic Wire-Harness
> Assembly: Paired Counterfactual Evaluation across Physics Backends**

最终标题仍需在主实验和外部基准复核完成后冻结。
