# RouteBot 直接 MuJoCo 闭环结果

## 证据边界

该实验每个控制步直接读取 MuJoCo 当前线缆、夹具、终点和任务空间夹爪位姿，
重建 29 节点关系图，再把冻结网络的输出映射回 MuJoCo mocap 末端目标。抓取、
入扣和终点事件只由 MuJoCo body 距离与原生 connect 约束触发，不读取 PBD
状态或 PBD 高层事件。

这仍不是关节力矩或末端力控制。高精 UR10e 和 Robotiq 网格是关闭碰撞的视觉
IK 孪生，因此结果只能称为跨物理后端任务空间闭环，不能称为真实机器人或产线
成功率。

## 教师可行性校准

最初的均匀弧长对应使 7/9 种语义关系对脚本教师也不可达。开发失败结果保留在
artifacts/papers/routebot/mujoco_direct_all9_pilot/。随后只用脚本教师校准夹具
路径和 PBD 粒子到 MuJoCo cable segment 的弧长锚点；没有以学习模型成绩选择
布局。校准后教师在九种关系上 9/9，通过后冻结场景和映射，再运行大样本比较。

冻结场景 SHA-256：

    52e1bbc7cd949e082ea977e05f15c041e0512a27e972fb0771095dadda95459a

完整锚点、仿射标定矩阵、每回合夹具扰动和选择规则均写入
artifacts/papers/routebot/final/mujoco_direct/route_mujoco_direct_report.json。

## 冻结协议

- 20 个配对物理 seed；
- 每个 seed 运行全部 9 种预声明线段—夹具关系；
- 每种关系中 TopoHarness、等参数 Geometry 与脚本教师共享相同物理扰动；
- 共 540 回合，每个策略 180 回合；
- 每动作 4 个 MuJoCo physics step，最多 220 个策略步；
- 成功要求三个语义线段全部入正确夹具，末端进入终点，且原生 XY 约束误差
  不超过 55 mm。

冻结权重 SHA-256：

- TopoHarness: 1c03f31147fc20c1668d29c1a024b30ede9c16419549583ce631e8f7bf659f69
- Geometry: 601f370b0ccfc284943457bea1673cf61a543392441fe81a9849a28ae5d46131

## 冻结结果

| 策略 | 回合 | 成功 | Wilson 95% CI | 九关系最小—最大 |
|---|---:|---:|---:|---:|
| TopoHarness | 180 | 180/180 (100%) | [97.9%, 100%] | 100%—100% |
| Geometry | 180 | 0/180 (0%) | [0%, 2.1%] | 0%—0% |
| Scripted teacher | 180 | 180/180 (100%) | [97.9%, 100%] | 100%—100% |

TopoHarness 与 Geometry 的 180 个配对回合全部不一致且全部有利于 TopoHarness。
双侧精确 McNemar 原始 p 值为 1.3051e-54，两项比较的 Holm 校正 p 值为
2.6101e-54，连续性修正匹配优势比为 361。

## 复现

    PYTHONPATH=src python -m harnessbench sim-eval-route-mujoco-direct \
      --physical-seeds 20 --assignment-indices 0 1 2 3 4 5 6 7 8 \
      --policies learned_topology learned_geometry teacher_topology \
      --physics-steps 4 --max-steps 220 --no-render \
      --output artifacts/papers/routebot/final/mujoco_direct

逐回合 CSV、按策略与关系汇总、精确配对检验、LaTeX 表格和机器可读报告均位于
artifacts/papers/routebot/final/mujoco_direct/。
