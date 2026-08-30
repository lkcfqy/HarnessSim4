# TopoHarness 开发结果

> 状态：PBD 闭环与分层 PBD→MuJoCo 开发证据，2026-08-26。不是冻结论文主表，
> 不代表端到端 MuJoCo 控制或真实产线性能。

## 可复现对象

- 数据：`artifacts/learning/topoharness_demos.npz`
- 数据 SHA-256：`f46e94e69b366d6e927abd444deb7587c3f18e128c6d962f9c94d824d254e1d4`
- 样本：10,542 帧；训练/验证/测试为 5,689/1,515/3,338 帧
- episode/任务：训练 24、验证 6、测试 12；整 episode seed 隔离
- Branch 语义置换：训练 12/12、验证 3/3、测试 6/6
- 模型：Topology 与 Geometry 均为 120,217 参数
- 检查点：`artifacts/learning/topology_policy.pt`、`geometry_policy.pt`
- Topology SHA-256：`2db1b1be45fff9617e3a028ecd70a1c02c18bbf0fc07df91adbb298df542d193`
- Geometry SHA-256：`cdb89dfb05196e8e11b76d6434807f3ebd035a9b8cca0e28b8fea1c860b2480f`

Topology 在第 119 epoch 取得最佳等任务权重验证指标，Geometry 在第 81 epoch。留出示教
帧上的总指针准确率分别为 98.18% 和 72.50%。离线指标仅用于诊断，主判断来自闭环。
RouteBot 的成功定义要求机器人显式抓取末端并完成终点放置，第三个卡扣闭合造成的被动
位移不能触发成功。

## 四任务闭环

每任务 × 3 难度 × 5 个未见 seed，共 180 个策略 episode；同一实验单元中的三个策略使用
同一环境 seed。完整逐 episode 记录位于 `artifacts/learning/evaluation/`。

| 机器人 | Learned Topology | Learned Geometry | Teacher Topology |
|---|---:|---:|---:|
| InspectBot | 14/15（93.3%） | 14/15（93.3%） | 15/15（100.0%） |
| InsertBot | 15/15（100.0%） | 15/15（100.0%） | 15/15（100.0%） |
| RouteBot | 13/15（86.7%） | 0/15（0.0%） | 12/15（80.0%） |
| BranchBot | 13/15（86.7%） | 0/15（0.0%） | 13/15（86.7%） |

Wilson 95% 区间、逐难度汇总和成对成功差见机器可读 CSV。Inspect 的 Geometry 更高，当前
不能主张四任务平均都受益于拓扑；Insert 是共同饱和任务。当前拓扑贡献主要由 Route 和
Branch 支持。

## Branch 语义反事实干预

30 个物理 seed（每难度 10 个）各运行 A/B 未置换与置换两种条件。每个策略每种条件 30
个 episode，总计 180 个策略 episode。两条件的物理几何、目标候选集合和随机种子相同；
只交换可观测语义关联。完整结果位于 `artifacts/learning/branch_counterfactual/`。

| 策略 | 未置换 | 置换 |
|---|---:|---:|
| Learned Topology | 26/30（86.7%） | 26/30（86.7%） |
| Learned Geometry | 0/30（0.0%） | 0/30（0.0%） |
| Teacher Topology | 26/30（86.7%） | 26/30（86.7%） |

该结果表明拓扑模型对语义重命名保持不变，而去除语义关系的模型不能解决该任务。它仍是
单一 PBD 环境中的开发实验；正式论文需扩大样本、加入配对置换检验和标准学习基线。

## 分层 PBD→MuJoCo 迁移审计

冻结 PBD 策略选择类型化目标与动作原语，MuJoCo 3.12 elasticity cable 场景执行三维动力学，
并使用原生检测覆盖、端点误差、卡扣语义/几何误差判定。每任务每策略运行 2 个开发 seed，
共 24 条回放，完整结果位于 `artifacts/learning/mujoco_transfer/`。

| 机器人 | Learned Topology | Learned Geometry | Teacher Topology |
|---|---:|---:|---:|
| InspectBot | 2/2 | 2/2 | 2/2 |
| InsertBot | 2/2 | 2/2 | 2/2 |
| RouteBot | 2/2 | 0/2 | 2/2 |
| BranchBot | 2/2 | 0/2 | 2/2 |

这是“PBD 闭合任务事件、MuJoCo 执行动作原语”的分层迁移，不是策略直接读取 MuJoCo 状态
并端到端控制。样本也远小于论文规模，不能据此声称 sim-to-real。

## 可视化

`artifacts/learning/visuals/harnesssim4_learned_topology_dashboard.html` 包含四个成功的冻结
模型闭环回放；同目录保存 GIF、最终帧和 manifest。视频不参与定量统计。

## 尚未满足的投稿门槛

- 每实验单元至少 100 个冻结测试 seed；
- ACT、Diffusion Policy 或等价标准基线；
- 学习式恢复/时序建模和低数据曲线；
- 每单元 100 seed 的端到端 MuJoCo 策略与另一独立仿真后端验证；
- 真实线材参数标定或最小实物验证。
