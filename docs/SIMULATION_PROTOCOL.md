# HarnessSim4 仿真与评测协议 v0.4

## 双后端路线

### 快速后端：当前可运行

二维 Position-Based Dynamics（PBD）用于批量训练、课程学习、奖励调试和拓扑消融：

- 粒子图支持链状和分支结构；
- 一阶边保持长度，二阶边近似弯曲刚度；
- 支持圆形障碍、边界、自碰撞、运动学抓持点和交叉计数；
- 全部随机数由 episode seed 控制；
- 无 GPU、无机器人和无原生仿真 DLL 也能运行。

### 高保真后端：已接入分层策略迁移，投稿前扩展

优先采用 MuJoCo 官方 elasticity cable plugin；它把不可伸长一维连续体离散化，面向杆件
扭转与弯曲。另选 Isaac Lab Newton/VBD cable backend 做 sim-to-sim 复核。高保真后端不
替代快速后端，而用于回答“结论是否依赖自研二维物理”的审稿问题。

仓库目前已有四个 MuJoCo 3.12 场景：Inspect/Insert/Route 各使用 1 个 cable plugin，
Branch 使用 trunk + 两个 branch 共 3 个 plugin，形成真实的 Y 型连接图。验证命令会：

1. 编译全部 MJCF；
2. 静止推进并检查 `qpos/qvel/qacc` 有限；
3. 运行脚本化 mocap 轨迹；
4. 记录抓持 equality anchor 的峰值和最终残差；
5. 输出 GIF、最终帧、contact sheet、交互面板和 JSON manifest。

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-mujoco-validate
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-mujoco-demo
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-mujoco-transfer --episodes 2
```

`sim-mujoco-transfer` 冻结 PBD 策略，由 PBD 闭合任务事件并选择类型化动作原语；MuJoCo
负责三维 cable dynamics，并以扫描覆盖、端点误差、卡扣语义与几何误差独立判定。当前
2 seed/任务/策略的 24 条审计用于验证接口和发现后端不一致，不是论文统计结果，也不是
策略直接读取 MuJoCo 状态的端到端闭环。

参考：

- [MuJoCo cable elasticity plugin](https://github.com/google-deepmind/mujoco/blob/main/plugin/elasticity/README.md)
- [MuJoCo repository and visualization](https://github.com/google-deepmind/mujoco)
- [Isaac Lab cable backend](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/using-cables.html)

## 难度参数

统一 `difficulty ∈ [0,1]`，但每个环境映射到有物理意义的随机化：

- InspectBot：位姿扰动、视野收缩、真阳性下降和假阳性上升；
- InsertBot：护套角度、通道间隙、线缆弯曲、障碍间隙和末端速度；
- RouteBot：卡扣偏移、障碍半径、线缆弯曲刚度和卡入公差；
- BranchBot：目标偏移、初始形状、分支扰动和双臂速度。

论文主实验固定 0.2、0.5、0.8 三档，连续难度只用于训练课程和敏感性曲线。

## 随机种子与成对设计

在同一任务和难度下，不同策略必须使用相同环境 seed。策略自身随机源使用由环境 seed
确定的独立偏移。这样 Topology 与 Geometry 的成功差可以做成对分析，不被不同初始场景
混淆。

默认开发实验每个单元 5 个 episode；论文主表每单元至少 100 个 episode。四任务×三难度
×三策略×100 seed 共 3,600 个 episode。

批量运行建议按 seed 使用独立进程；进程池保持输入顺序，因此同一配置的逐 episode 记录
顺序不随 worker 数改变：

```powershell
python -m harnessbench sim-benchmark --episodes 100 --workers 4 --output artifacts/paper_main
```

## 学习数据与反捷径设计

- 示教以完整 episode seed 为最小切分单位，连续状态不得跨训练/验证/测试；
- 训练采样按任务等权，不让长轨迹任务主导早停；
- 二值动作阈值只允许在验证集校准，测试集不能参与模型或阈值选择；
- Geometry 消融和 Topology 模型参数量均为 120,217，使用同样本、优化器和日程；
- Geometry 消融移除物理/语义边、端点/分叉角色、A/B 身份和工序顺序；
- BranchBot 每个数据切分内平衡 A/B 语义重命名，两种条件保留同一几何候选集合；
- 精确干预评测对同一物理 seed 强制运行两种命名，避免把布局差异误认为拓扑增益。
- RouteBot 只有在机器人显式抓持末端并完成终点放置后才成功；卡扣闭合造成的被动位移
  不能提前终止 episode。

```bash
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-generate-demos \
  --train-episodes 24 --validation-episodes 6 --test-episodes 12
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-policy --mode both \
  --epochs 120 --learning-rate 0.0005 --hidden-dim 64 --patience 25
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-learned --episodes 5
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-eval-branch-counterfactual --episodes 10
```

上述 5/10 均为开发规模。正式论文在代码、模型、阈值和 seed 清单冻结后，每个主实验单元
至少运行 100 个 episode，并另外报告 A/B 两种干预条件。

## 统计报告

- 成功率：Wilson 95% 置信区间；
- 连续指标：均值与 95% 区间，并保留逐 episode CSV；
- 主比较：同 seed 的 `Topology − Geometry` 成功差；
- 正式论文增加配对置换检验、效应量和多重比较校正；
- 同时报告平均/P90/P99 步数、失败类型和计算耗时；
- 不删除失败 seed，不根据测试集调阈值。

## 物理验证门槛

二维后端至少通过：

1. 静止直线在无外力时保持；
2. 固定端位移后长度误差有界并收敛；
3. 障碍穿透率低于预设阈值；
4. 已知两线段交叉计数正确；
5. 相同 seed 逐步结果确定；
6. 时间步、子步数和约束迭代数敏感性分析。

高保真后端当前已通过四场景编译、静止步进、脚本轨迹检查和 24 条分层策略迁移开发审计。
正式实验仍须使用端到端 MuJoCo 观测策略复现每个任务至少 100 个冻结 seed，并报告策略
排名相关性、成功判定一致率和失败类型。

## 可视化验收

每个任务必须输出：初始帧、关键中间帧、最终帧、循环 GIF，以及能播放、暂停和逐帧拖动
的自包含面板。视频只用于解释行为；定量结论必须来自机器可读结果。

冻结学习模型的可视化命令：

```bash
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-demo-learned
```
