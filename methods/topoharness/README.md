# TopoHarness Policy

四任务共享的可训练图指针策略。节点表示线束粒子、端点、分叉、检测位、连接器、卡扣、
工艺目标和机器人末端；物理边与工艺关联边经三层消息传递后，四个动作头输出连续动作，
两个指针头选择操作节点。类型约束解码器把指针落到可执行的检测、插接、抓线、卡扣或双臂
解缠航点上，抓取/扫描时机只使用可观测距离、状态和置信度。

公平的 Geometry 消融保持相同的 120,217 个参数、训练样本和优化日程，但关闭图边、端点/
分叉角色、A/B 身份和工序顺序。BranchBot 在每个数据切分中按 1:1 随机重命名 A/B 分支，
两种命名具有相同几何候选集合，防止模型靠固定坐标背答案。数据按完整 episode seed 划分，
相邻帧不会跨训练、验证和测试集；二值阈值只在验证集校准。

WSL CPU 运行：

```bash
.venv-wsl/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-generate-demos \
  --train-episodes 24 --validation-episodes 6 --test-episodes 12
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-policy --mode both \
  --epochs 120 --learning-rate 0.0005 --hidden-dim 64 --patience 25
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-learned --episodes 5
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-eval-branch-counterfactual --episodes 10
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-demo-learned
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-mujoco-transfer --episodes 2
```

当前 180 条留出 seed 的 PBD 开发评测中，学习拓扑版/无拓扑版成功率分别为：Inspect
93.3%/93.3%、Insert 100%/100%、Route 86.7%/0%、Branch 86.7%/0%。在 30 个物理
seed × 两种 A/B 命名的精确干预中，Branch 拓扑版两组均为 86.7%，几何版两组均为 0%。
2 seed/任务/策略的分层 PBD→MuJoCo 审计中，拓扑版四任务均为 2/2；几何版在 Route/Branch
均为 0/2。以上是开发结果，不是冻结论文主表；样本量仍小，Inspect 也不应宣称拓扑优势。

离线 action error 只用于调试；论文主指标必须是留出 seed 的闭环成功率。正式投稿仍需
100 seed/单元、标准 ACT/Diffusion 基线、失败分类，以及端到端 MuJoCo/第三方后端复核。
