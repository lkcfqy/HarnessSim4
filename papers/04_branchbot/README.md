# BranchBot research draft

论文题目：*Semantic Counterfactuals Expose Branch-Identity Shortcuts in Bimanual
Wire-Harness Manipulation*。

## 当前结论

物理初态、夹具、动力学与随机种子固定，仅交换可观测 A/B 身份绑定。TopoHarness 在 600 条
正式 PBD episode 中成功 480 次，Geometry 成功 123 次；Geometry 的成功全部在 canonical
条件，swap 后为 0/300。ACT direct 为 432/600，而加入类型化指针或关系池化后分别达到
478/600 与 479/600。三个图消融也都在 79.5%--80.2%。因此论文结论是语义通道与类型化
动作落地必要、具体架构不唯一。

直接 MuJoCo 复核使用当前 cable/site/mocap 位姿重建观测并闭环执行两臂命令，不回放 PBD
轨迹：TopoHarness 40/40，Geometry 13/40；Geometry 在 swap 条件为 0/20。

## 构建

从仓库根目录、WSL Ubuntu 环境运行：

```bash
scripts/build_branchbot_research_draft.sh
```

完整正式评测（会重算 6,120 条闭环 episode）：

```bash
scripts/run_branchbot_formal.sh
```

输出 PDF：`output/pdf/04_branchbot_research_draft.pdf`。

核心机器可读证据：

- `artifacts/papers/branchbot/final/counterfactual/branch_paper_report.json`
- `artifacts/papers/branchbot/final/mujoco_direct/branch_mujoco_direct_report.json`
- `artifacts/papers/branchbot/final/audit/branchbot_result_audit.json`
- `artifacts/papers/branchbot/final/branchbot_research_draft.sha256`

## 声明边界

这不是投稿终稿，也不是实体机器人结果。当前仍缺少 3--5 分支组合泛化、原始图像分支身份
识别、遮挡与非对称材料、关节/力控仿真、实体双臂台架以及多训练种子不确定性。正文保留
`DRAFT---NOT FOR SUBMISSION` 标记。
