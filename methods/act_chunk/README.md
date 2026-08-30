# ACT-style Action Chunking Baseline

本目录记录 HarnessSim4 的低维状态 ACT 基线。实现位于
`src/harnessbench/learning/act_baseline.py`。

核心结构遵循 ACT 的研究思路：条件变分编码器、Transformer action queries、一次预测固定
长度动作块，以及闭环时间集成。输入是与 TopoHarness 相同的可观测 typed-node features 和
全局状态。`ACT-Geometry` 不使用邻接矩阵；更强的 `ACT-RelPool` 接收同一语义邻接矩阵，
但只做固定的一跳邻居特征池化，不使用 TopoHarness 的可学习图消息传递。

这不是官方 ACT 代码的拷贝或精确复现。与原论文最重要的差异是：

- 原 ACT 使用多视角图像和机器人关节状态；本实现使用 HarnessSim4 低维节点集合；
- 原任务是实体 ALOHA 双臂精细操作；本实现评测 PBD 线束任务；
- 本实现保留 CVAE、动作块和时间集成，但模型规模缩小到可在 CPU 上复现；
- 推理时同原 ACT 思路使用潜变量先验均值，不随机采样动作风格。

参考：

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*,
  RSS 2023, <https://arxiv.org/abs/2304.13705>；
- 官方实现：<https://github.com/tonyzhaozh/act>。

RouteBot 配置：

```bash
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-act \
  --tasks route --epochs 80 --batch-size 256 --learning-rate 0.002 \
  --hidden-dim 48 --heads 4 --layers 1 --chunk-size 8 --latent-dim 8 \
  --kl-weight 0.1 --patience 12 \
  --output artifacts/learning/act_chunk_policy.pt

PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-act \
  --tasks route --use-relations --epochs 80 --batch-size 256 --learning-rate 0.002 \
  --hidden-dim 48 --heads 4 --layers 1 --chunk-size 8 --latent-dim 8 \
  --output artifacts/learning/act_relational_policy.pt
```

配对闭环主表：

```bash
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-paper \
  --episodes 100 --difficulties 0.2 0.5 0.8 --workers 8 \
  --act-relational-checkpoint artifacts/learning/act_relational_policy.pt \
  --output artifacts/papers/routebot/main_table

PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-counterfactual \
  --physical-seeds 100 --workers 8 \
  --act-relational-checkpoint artifacts/learning/act_relational_policy.pt \
  --output artifacts/papers/routebot/counterfactual
```

评测同时报告两个版本：`act_chunk` 直接执行连续动作；`act_chunk_typed` 使用同时训练的
Transformer 指针头和与主方法同类的候选类型过滤，选择可执行线段/夹具。两者共享同一个
检查点。前者是标准动作块基线，后者用于排除“性能差异仅来自离散动作接口”的解释；后者
仍不使用图边或图消息传递。另报告 `act_relational_typed`：它使用固定关系池化与同一指针
接口，用于区分“显式关系输入”的收益和“可学习图消息传递”的额外收益。

训练报告会保存数据哈希、完整配置、参数量、阈值校准、学习曲线和上述实现边界。论文中应写
“ACT-style state baseline”，除非未来严格对齐官方图像/关节输入和训练协议后重新验证。
