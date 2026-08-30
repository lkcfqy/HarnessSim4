# InspectBot：拓扑风险主动检测机器人

这是四机器人计划的第一个公开数据研究项目。系统把证据严格分成三层：

1. **真实公开图像**：MVTec AD `cable` 与 FAU/FAPS Stripped Wire；
2. **混合主动检测**：把冻结的真实图像异常分数嵌入 12 个线束检测位；
3. **3-D 执行审计**：MuJoCo 中的 UR5e + 双镜头检测头执行相同扫描决策。

它不是实体机器人实验，也没有把 MuJoCo 渲染图送进异常检测器。

## 当前冻结结果

| 证据 | 规模 | 主要结果 |
|---|---:|---|
| MVTec cable | 224 train-good；150 test | Spatial Gaussian image AUROC 92.2%，pixel AUROC 95.7% |
| Stripped Wire | 200 train-good；300 test | Spatial NN AUROC 81.5%；正常样本阈值下 recall 41.3% |
| adverse-first stress-v1 | 1,000 paired scenarios | 遮挡暗光视图 recall 1.1%，策略优势被感知下限压没；负结果保留 |
| clean-first deployment-v2 | 2,000 paired scenarios | Topology-risk 相对 Geometry 在预算 4/6/8/10 提升 9.4/7.8/7.7/4.7 个百分点 |
| direct MuJoCo | 20 seeds × 4 policies × 4 budgets | 320/320 到位；FOV 误差 ≤0.97 mm；视觉 IK 误差 ≤1.50 mm |

所有主动检测比较均按相同 seed 配对。部署协议使用 10,000 次场景 bootstrap，且把全部
30 个 topology-vs-baseline Wilcoxon 检验一起做 Holm 校正。描述性 oracle 的原始实现错误被
保留，修正后 72,000 条非 oracle 行逐字段不变。

## 公开数据与许可

- MVTec AD cable：CC BY-NC-SA 4.0；本地树哈希与镜像 commit 记录在
  `docs/INSPECTBOT_LITERATURE_AND_DATA_AUDIT.md`。
- Stripped Wire v2：CC BY 4.0，DOI `10.5281/zenodo.16686806`；原压缩包 MD5、SHA-256、
  train/test 重复检查均已冻结。
- CableInspect-AD 只完成官方论文和代码审计。官方数据端点需要访问批准，因此没有下载、
  没有实验数字，也没有替换成来源不明的镜像。

## 复现入口

在 WSL 的项目根目录运行：

```bash
# 独立结果审计（787 checks）
PYTHONPATH=src .venv-dlolab/bin/python scripts/audit_inspectbot_results.py

# 重新生成论文图表、表格、清单
PYTHONPATH=src .venv-dlolab/bin/python scripts/render_inspect_paper_assets.py

# 测试、审计、图表和英文 PDF 一键构建
bash scripts/build_inspectbot_research_draft.sh
```

发布草稿：`output/pdf/01_inspectbot_research_draft.pdf`。

## 场景与旧统一接口

基础 PBD 配置见 `configs/sim/inspect.json`，环境实现在
`src/harnessbench/sim/envs/inspect.py`：

```powershell
python -m harnessbench sim-demo --tasks inspect
python -m harnessbench sim-benchmark --tasks inspect --episodes 20
```

`mujoco/scene.xml` 和运行时资产构成 UR5e + 双镜头检测头视觉孪生，包含检测夹具、状态灯、
安全护栏和 12 个风险检测位。机器人网格碰撞关闭，不改变线束物理。

## 尚不能声称的内容

- 没有真实机械臂、工业相机、标定、节拍、碰撞安全或跨工厂证据；
- 四种 appearance 是确定性图像扰动，不是真实多视角轨迹；
- 风险权重由工程先验给定，不是模型学习结果；
- 论文是研究草稿，不是可直接投稿的终稿。实体台架、真实相机闭环、跨供应商测试和多训练
  seed 仍是升级为投稿终稿的必要工作。
