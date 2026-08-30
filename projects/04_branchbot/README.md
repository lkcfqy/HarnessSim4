# BranchBot

双臂多分支线束解缠与布置机器人。配置见 `configs/sim/branch.json`，环境实现在
`src/harnessbench/sim/envs/branch.py`。

当前研究问题：在物理几何完全不变、仅交换可观测 A/B 端点—机械臂—目标绑定时，策略能否
保持正确分配。现有证据支持“语义身份必须被表示并落地到类型化动作”，但不支持“某一种
GNN 架构独有优势”。

```powershell
python -m harnessbench sim-demo --tasks branch
python -m harnessbench sim-benchmark --tasks branch --episodes 20
```

正式研究包包含：

- 420 条原始专家轨迹、280 条成功且平衡的冻结 train/val/test 轨迹；
- 7 个同数据冻结学习模型；
- 100 物理种子 × 3 难度 × 2 语义指派 × 10 策略，共 6,000 条 PBD 闭环 episode；
- 20 物理种子 × 2 指派 × 3 策略，共 120 条直接观测 MuJoCo 闭环 episode；
- 精确 McNemar 检验、Holm 校正、语义不变性、失败分类、182 项独立结果审计；
- 4 页英文研究草稿、自动生成表格/热图和一键复现脚本。

在 WSL Ubuntu 环境运行：

```bash
scripts/run_branchbot_formal.sh
scripts/build_branchbot_research_draft.sh
```

冻结主结果：TopoHarness 480/600，Geometry 123/600，ACT direct 432/600，
ACT-Pointer 478/600，ACT-RelPool 479/600，教师 478/600。Geometry 的 123 次成功全部来自
canonical 指派，A/B swap 后为 0/300。直接 MuJoCo 中 TopoHarness 40/40，Geometry
13/40，且 Geometry swap 后为 0/20。

真实可视化：`mujoco/scene.xml` 与运行时拼装的双 UR10e + 双 Robotiq 2F-85 视觉孪生，
用三个 elasticity cable plugin 实例组成 trunk + A/B 两分支，并配置双夹爪、语义目标、
分离夹具、安全护栏、工厂灯光和双视角相机。机器人网格碰撞关闭，不改变线束物理；边界见
`configs/sim/realistic_visuals.json`。

证据边界：当前固定为两分支、状态观测、PBD 与 task-space mocap MuJoCo；尚未验证 3--5
分支、图像感知、力/力矩控制或真实产线。论文因此明确标为 research draft。
