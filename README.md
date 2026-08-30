# HarnessSim4

面向汽车线束智能制造的纯仿真、可视化、拓扑感知机器人研究平台。

当前已形成 InspectBot、InsertBot、RouteBot 与 BranchBot 四套可复核研究草稿；它们都明确标注为
`DRAFT—NOT FOR SUBMISSION`，不能替代实体机器人实验。InspectBot 已完成两套真实公开图像、
成对主动检测和 320 次直接 MuJoCo 扫描审计，详见
[`projects/01_inspectbot/README.md`](projects/01_inspectbot/README.md)。RouteBot 的公开数据、
配对反事实、MuJoCo 与 DLO-Lab 外部复核范围见
[`docs/ROUTEBOT_PROJECT_CHARTER.md`](docs/ROUTEBOT_PROJECT_CHARTER.md)。四机器人编号沿用仓库
原始目录。

项目把一条线束工艺链拆成四个可独立训练、又共享状态表示和评测协议的机器人任务：

| 子项目 | 机器人 | 仿真任务 | 核心科学问题 |
|---|---|---|---|
| `01_inspectbot` | InspectBot | 主动扫描端子、颜色、顺序、卡扣与缺陷 | 拓扑引导的主动感知与异常检测 |
| `02_insertbot` | InsertBot | 端子对准、插壳、接触力代理与锁止 | 柔性扰动下的视觉—力觉精密插入 |
| `03_routebot` | RouteBot | 抓取语义线段并依次压入卡扣 | DLO路径控制与拓扑对应关系 |
| `04_branchbot` | BranchBot | 双臂分离、解交叉并布置多分支线束 | 分支身份保持、解缠与双臂协同 |

四个任务统一构成拟投稿 Benchmark：

> **HarnessSim4: A Topology-Aware Benchmark for Manipulation of Branched
> Deformable Linear Objects in Wire-Harness Assembly**

## 已实现

- 纯 NumPy 的二维 Position-Based Dynamics 柔性线束快速后端；
- 链状和多分支图、拉伸/弯曲约束、障碍碰撞、自碰撞与交叉计数；
- 四个 Gym 风格环境：`reset / step / observation / reward / termination / info`；
- Topology、Geometry、Random 三组可复现实验基线；
- 共享的 TopoHarness 图指针学习策略、同参数量无拓扑消融和类型约束动作解码；
- 整 episode 数据隔离、任务平衡采样、仅验证集阈值校准及 A/B 语义反捷径干预；
- 四个可编译的 MuJoCo 3.12 三维场景，均使用官方 elasticity cable plugin；
- UR5e、KUKA iiwa 14、UR10e 与 Robotiq 2F-85 高精网格组成的真实机器人视觉孪生；
- 统一工厂工位、阴影/安全护栏、主视角 + 工艺特写和 960×540 双视角动画；
- 1920×1080、24 fps、35.875 秒的四机器人统一 H.264 影片，含六个章节、交叉转场、
  同步四工位结尾、交互式章节跳转和 24 项最终编码审计；
- 冻结策略的类型化动作原语可迁移至 MuJoCo，并由三维原生几何条件独立判定；
- WSL/EGL 无头渲染、有限状态检查、约束残差与高保真 GIF/交互回放；
- 成对随机种子、三级难度、Wilson 成功率区间和连续指标置信区间；
- GIF、最终帧与可播放/暂停/逐帧拖动的交互式回放面板；
- 原始 episode、汇总表和拓扑方法相对几何方法的配对结果，输出 JSON/CSV；
- LeRobot ALOHA 公开插入代理数据已完成 50 条完整 episode、25,000 帧、40/10 episode 留出
  的离线动作审计，并作为任务不匹配的辅助证据纳入 InsertBot 研究稿；
- Berkeley Cable Routing 的 1,647 条真实机器人轨迹、42,328 帧及四路相机流已固定版本并完成逐文件 SHA-256 校验；
- 公开 RouteBot 数据按完整 episode 划分 1,153/247/247，已完成动作持久性与时序岭回归的配对离线审计；
- RouteBot 已加入直接 MuJoCo 闭环：每步从当前三维线缆/夹具位姿重建图观测，
  冻结策略直接输出任务空间动作，并由原生 cable 约束判定入扣与终点；
- 独立实现的低维状态 ACT-style CVAE/Transformer 动作块基线与 RouteBot 配对检验。
- InspectBot 已审计 MVTec AD cable 与 FAU/FAPS Stripped Wire 两套公开图像；
- InspectBot 已冻结 adverse-first 负结果与 2,000-scenario clean-first 配对主表，描述性 oracle
  修正前后的 72,000 条非 oracle 结果逐字段一致；
- InspectBot 直接 MuJoCo 扫描 320/320 到位，独立结果审计 787 项全过；
- InsertBot 已完成 25,000 帧公开代理数据审计、4,800 条冻结成对接触基准、300 条 direct
  MuJoCo 闭环审计和 153/153 项独立结果检查；hard 条件下相对直线插入的成功率提高
  33.5 个百分点，但未显著
  超过 guarded admittance，因此保留强基线反证与硬件门槛；
- 四份英文研究草稿 PDF 位于 `output/pdf/`，均保留仿真/公开数据/硬件证据边界。
- 四份 PDF 的页面几何、字体嵌入、LaTeX 错误、未解析引用和 Overfull 溢出已进入机器审计；
  当前四篇均通过这些排版完整性门槛。
- 四篇均已转换为匿名 `IEEEtran` Transactions 双栏格式并按目标期刊分流：InspectBot 与
  InsertBot 面向 T-ASE，RouteBot 与 BranchBot 面向 T-RO；四套匿名暂存投稿包、独立 H.264
  补充视频、ReadMe、摘要、cover letter 和清单均已生成，期刊格式门禁为 `4/4`。
- 四项实体研究的预注册数据字段、标定/来源/媒体证据和成对统计门禁已机器化；当前没有
  实体试验数据，硬件门槛仍为 `0/4`。

当前快速后端用于算法迭代和消融，不代表真实物理。学习策略已经在四任务 PBD 闭环中运行，
并完成首轮分层 PBD→MuJoCo 策略迁移审计；RouteBot 另有不读取 PBD 状态或事件的直接
MuJoCo 观测—任务空间动作闭环。后者仍使用 mocap 末端代理和关闭碰撞的视觉 IK 机械臂，
不是力矩/力控制，也不是硬件证据。SCI 投稿前仍须按
[`docs/SCI_PAPER_PLAN.md`](docs/SCI_PAPER_PLAN.md) 扩大冻结测试、增加标准学习基线，并完成
端到端 MuJoCo 与第三方后端复核。

## 运行

本 GitHub 仓库保存可复现源码、文档与轻量研究材料，不作为完整数据备份。公开数据、
生成产物、本地虚拟环境和匿名投稿暂存包仍保存在完整 U 盘快照中，不进入普通 Git 历史。
首次克隆时请同时初始化固定版本的外部代码：

```bash
git clone --recurse-submodules https://github.com/lkcfqy/HarnessSim4.git
```

Windows PowerShell：

```powershell
python -m pip install -e .
python -m harnessbench sim-list
python -m harnessbench sim-demo --tasks all --policy topology
python -m harnessbench sim-benchmark --episodes 5 --workers 4
```

本机 Windows 的应用控制会阻止 MuJoCo 原生 DLL，仓库已验证的高保真运行方式是 WSL2：

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/pip install -e '.[dev,high-fidelity,learning]'
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-mujoco-validate
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-mujoco-demo
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-generate-demos \
  --train-episodes 24 --validation-episodes 6 --test-episodes 12
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-policy --mode both \
  --epochs 120 --learning-rate 0.0005 --hidden-dim 64 --patience 25
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-learned --episodes 5
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-act --tasks route
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-paper --episodes 100
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-counterfactual \
  --physical-seeds 10
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-download-route-public \
  --include-videos
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-public
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-render-route-public
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-eval-route-mujoco-direct --physical-seeds 2
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-demo-learned
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-mujoco-transfer --episodes 2
```

主要输出：

- `artifacts/simulations/harnesssim4_dashboard.html`：四机器人交互回放；
- `artifacts/simulations/*.gif`：四个可循环动画；
- `artifacts/sim_benchmark/benchmark_report.json`：完整实验与运行环境；
- `artifacts/sim_benchmark/episodes.csv`：逐 episode 数据；
- `artifacts/sim_benchmark/summary.csv`：按任务/策略/难度汇总；
- `artifacts/sim_benchmark/paired_topology_advantage.csv`：成对比较。
- `artifacts/mujoco_validation/mujoco_hifi_dashboard.html`：四场景 MuJoCo 回放；
- `artifacts/mujoco_validation/mujoco_manifest.json`：模型规模、稳定性与约束误差。
- `artifacts/showcase/harnesssim4_four_robot_showcase.mp4`：四机器人统一电影级演示；
- `artifacts/showcase/harnesssim4_four_robot_showcase.html`：可播放、拖动和章节跳转的演示页；
- `artifacts/showcase/harnesssim4_four_robot_media_audit.json`：最终 H.264 反向解码审计。
- `artifacts/submission_bundle/four_paper_readiness.json`：四篇论文的机器可读投稿门禁；
- `artifacts/submission_bundle/FOUR_PAPER_READINESS.md`：逐篇研究稿、结果审计和硬件门槛汇总。
- `artifacts/submission_bundle/venue_package_audit.json`：目标期刊格式、摘要、NtP、关键词、
  PDF、补充视频、匿名包和严格硬件状态的机器审计；
- `submissions/01_inspectbot_tase/` 至 `submissions/04_branchbot_tro/`：四套明确标为
  `STAGING ONLY — DO NOT UPLOAD` 的匿名投稿暂存包。
- `artifacts/learning/evaluation/`：180 条学习策略闭环开发评测及置信区间；
- `artifacts/learning/branch_counterfactual/`：同物理 seed 的 A/B 语义配对干预；
- `artifacts/learning/visuals/harnesssim4_learned_topology_dashboard.html`：冻结模型四任务回放。
- `artifacts/learning/mujoco_transfer/`：24 条分层策略迁移、MuJoCo 原生判定与最终帧。
- `artifacts/papers/routebot/main_table/`：RouteBot 的 ACT 配对主表、精确检验和失败分类。
- `artifacts/papers/routebot/counterfactual/`：同物理 seed 的九种经独立教师语义完成审计并冻结的线段—卡扣绑定干预。
- `artifacts/papers/routebot/public_real/`：247 条留出真实机器人轨迹的离线时序动作审计、配对统计与论文表格。
- `artifacts/papers/inspectbot/final/`：公开图像感知、四视图 score cache、两版主动检测、
  direct MuJoCo、787 项审计和论文图表。
- `artifacts/papers/insertbot/formal/`：公开代理数据复核、4,800 条成对接触实验、300 条
  direct MuJoCo episode、统计检验、153 项独立审计与论文图表。
- `output/pdf/01_inspectbot_research_draft.pdf`：InspectBot 英文研究草稿。
- `output/pdf/02_insertbot_research_draft.pdf`：InsertBot 英文研究草稿。
- `output/pdf/03_routebot_research_draft.pdf`：RouteBot 英文研究草稿。
- `output/pdf/04_branchbot_research_draft.pdf`：BranchBot 英文研究草稿。

论文级全量实验建议每个实验单元至少 100 个种子：

```powershell
python -m harnessbench sim-benchmark --episodes 100 --workers 4 --output artifacts/paper_main
```

## 目录

```text
configs/sim/                  四任务与统一 Benchmark 配置
projects/01_inspectbot/       主动检测任务
projects/02_insertbot/        端子插入任务
projects/03_routebot/         入卡扣布线任务
projects/04_branchbot/        多分支双臂任务
projects/*/mujoco/scene.xml   四个三维弹性线缆场景
src/harnessbench/sim/         物理、环境、策略、统计和可视化
src/harnessbench/learning/    图编码、数据、训练、闭环策略、评测和学习策略可视化
methods/topoharness/          学习方法与复现实验说明
methods/act_chunk/            ACT-style 动作块基线及与原实现的差异
third_party/mujoco_menagerie/ 保留许可证的机器人/夹爪网格资产
tests/                        单元、四任务闭环与导出测试
web/dashboard_template.html   交互回放模板
docs/FOUR_ROBOTS.md           四机器人任务边界
docs/FOUR_ROBOT_SHOWCASE.md   统一电影级演示、生成命令和媒体证据边界
docs/HARDWARE_EVIDENCE_GATE.md 四项实体研究的数据结构、指标和可执行门禁
docs/SIMULATION_PROTOCOL.md    仿真与评测协议
docs/SCI_PAPER_PLAN.md         SCI论文假设、实验矩阵和投稿门槛
docs/SUBMISSION_READINESS.md   四篇论文逐项终稿审计与冻结执行顺序
docs/TOPOHARNESS_RESULTS.md     学习策略开发结果、哈希、边界与失败说明
docs/ROUTEBOT_PUBLIC_DATA_RESULTS.md 公开真实机器人线缆布线数据、冻结划分与配对结果
```

## 旧版公开数据基线

公开插入示教管线仍可运行：

```powershell
python -m harnessbench download-public
python -m harnessbench train-public
```

它使用 MIT 许可证的
[`lerobot/aloha_sim_insertion_human`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)，
并严格按 episode 划分。结果见 [`docs/BASELINE_RESULTS.md`](docs/BASELINE_RESULTS.md)。

## 研究边界

- 二维 PBD 的“成功率”只能比较仿真策略，不能声称真实线束装配成功；
- 启发式策略只作为教师/诊断基线；当前学习主线是 TopoHarness 图指针策略；
- 四任务 MuJoCo 分层迁移仍依赖 PBD 高层事件；RouteBot 已额外实现直接 MuJoCo 位姿观测
  与任务空间闭环，但没有真实线材参数标定，也不是关节力矩/末端力控制；
- 真实机器人网格目前由位置 IK 驱动且关闭碰撞，只是视觉孪生，不是力矩控制机器人动力学；
- 所有结果需报告随机种子、难度、失败类型和置信区间，不能只展示成功视频。

四篇论文的当前门禁可重复执行：

```powershell
python scripts/audit_submission_bundle.py
python scripts/build_submission_packages.py --skip-video
python scripts/audit_venue_packages.py --require-format-ready
python scripts/audit_submission_bundle.py --require-submission-ready
python scripts/audit_venue_packages.py --require-submission-ready
```

研究稿完整性和目标期刊格式当前均为 `4/4`。两个 `--require-submission-ready` 命令是严格
投稿门禁，在任一实体硬件证据包未通过时以状态码 2 退出。当前实体硬件门槛为 `0/4`，因此
不能删掉论文中的 `DRAFT—NOT FOR SUBMISSION` 标记，也不能上传 `submissions/` 中的暂存包。
目标期刊选择和官方规则冻结记录见
[`docs/VENUE_AND_SUBMISSION_STRATEGY.md`](docs/VENUE_AND_SUBMISSION_STRATEGY.md)。

实体台架准备完成后，用 `scripts/hardware_study.py` 执行初始化、协议冻结、标定登记、逐次
采集、媒体哈希登记和完成封存，再用 `scripts/audit_hardware_gate.py` 生成逐机器人硬件门禁；
字段与采集顺序见
[`docs/HARDWARE_EVIDENCE_GATE.md`](docs/HARDWARE_EVIDENCE_GATE.md)。
