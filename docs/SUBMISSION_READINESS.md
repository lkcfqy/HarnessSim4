# HarnessSim4 四篇论文投稿终稿审计

更新日期：2026-08-27

## “投稿终稿”的统一判定

本项目只有在下列项目全部满足后，才把某一机器人标记为 `submission-ready`：

1. 英文 LaTeX 论文可无错误编译，包含摘要、相关工作、方法、实验、局限、伦理与参考文献；
2. 核心主张对应预先写明的可证伪假设，而不是只展示动画或成功案例；
3. 冻结测试集每个主实验单元至少 100 个配对 seed，并报告 95% 置信区间、效应量和配对检验；
4. 至少包含 Geometry、Random/Scripted 和一个公认时序模仿学习基线；
5. 包含表示、数据量、难度和关键模块消融，以及失败类型与计算开销；
6. 至少在 MuJoCo 中进行策略级复核，明确二维 PBD 与三维仿真的职责边界；
7. 匿名复现包能从锁定配置、模型权重和 seed 重新生成论文表格与图片；
8. 不把纯仿真结果表述为真实产线性能，真实机器人视觉模型也不冒充力矩控制仿真。

当前没有任何一篇达到上述终稿标准。四篇均已具备可运行任务、学习方法雏形和真实机器人
数字孪生视觉基础；RouteBot 与 BranchBot 的科学信号最强，应先完成。

## 可执行投稿门禁

仓库不再只靠文字判断“终稿”。下列审计会复核四篇 LaTeX 源稿与发布 PDF 的哈希一致性、
页数、摘要、参考文献、局限、`DRAFT` 声明、逐项目结果审计，以及实体硬件证据包：

```powershell
python scripts/audit_submission_bundle.py
python scripts/audit_submission_bundle.py --require-submission-ready
```

输出位于 `artifacts/submission_bundle/four_paper_readiness.json` 和
`artifacts/submission_bundle/FOUR_PAPER_READINESS.md`。截至 2026-08-27，四篇研究稿完整性
均通过（`4/4`），但四项实体硬件门槛均未通过（`0/4`）；严格命令必须以状态码 2 拒绝
“可投稿终稿”结论。只有相应 `artifacts/hardware/<robot>/hardware_gate_audit.json` 通过，且
论文删除草稿警告并重新审计后，某篇论文才可标记为 `submission-ready`。

四项门禁现在均有冻结 JSON 模板和共同审计器。所需逐试验字段、配对设计、标定有效期、
来源签名、原始媒体哈希和成功状态不变量见 `docs/HARDWARE_EVIDENCE_GATE.md`；初始化器不会
覆盖既有采集文件，审计器会在缺失或矛盾证据上失败关闭。

## 目标期刊与投稿暂存包

当前冻结分流为 InspectBot、InsertBot 面向 IEEE T-ASE，RouteBot、BranchBot 面向 IEEE
T-RO。四篇均已使用匿名 IEEE Transactions 双栏版式；摘要均不超过 200 词，T-ASE 两篇
均含 100--300 词 Note to Practitioners，当前页面数依次为 6、6、5、4。官方规则、选择理由
和声明边界见 `docs/VENUE_AND_SUBMISSION_STRATEGY.md`。

四套暂存包位于 `submissions/`，分别包含与发布稿哈希一致的 PDF、匿名 cover letter、投稿
清单、6.25 秒项目专属 H.264 数字孪生视频、ReadMe、五句以内 Summary、声明边界和逐文件
SHA-256 manifest。下列审计把“期刊格式合格”和“科学证据可投稿”分开：

```powershell
python scripts/build_submission_packages.py --skip-video
python scripts/audit_venue_packages.py --require-format-ready
python scripts/audit_venue_packages.py --require-submission-ready
```

截至 2026-08-27，`venue_format_ready=4/4`，但严格命令仍以状态码 2 拒绝通过，因为硬件门槛
为 `0/4` 且四篇正确保留草稿警告。暂存包明确写有 `STAGING ONLY — DO NOT UPLOAD`。

## 已完成的共同基础

- 四任务统一 PBD API、示教生成、TopoHarnessNet 和 Geometry 消融；
- 四个 MuJoCo elasticity 场景与三维原生成功判据；
- UR5e、KUKA iiwa 14、UR10e、Robotiq 2F-85 的保留许可证网格资产；
- 960×540、72 帧、主视角 + 工艺特写的统一工厂渲染；
- 1920×1080、24 fps、35.875 秒的四机器人统一 H.264 影片与章节式交互页面；
- 最终视频已从 H.264 反向解码审计，尺寸、帧率、861 帧、六个章节与输入/输出哈希共
  24/24 项通过；
- 视觉机器人用位置逆解追随任务目标，机器人和夹爪碰撞关闭，不改变线束基准物理；
- Windows 与 Linux/MuJoCo 自动测试，以及逐 episode CSV/JSON 结果导出。
- 四份发布 PDF 均为一致的 US Letter 页面，字体全部嵌入；最终 LaTeX 日志无编译错误、
  未解析引用或 Overfull 溢出。RouteBot 热力图的未嵌入字体已在 2026-08-27 修复并重编。

本次正式视觉验证：

| 项目 | 机器人 | 网格数 | 视觉 IK 峰值 | 最终夹持误差 |
|---|---|---:|---:|---:|
| InspectBot | UR5e + 双目检测头 | 20 | 1.49 mm | 0.00 mm |
| InsertBot | KUKA iiwa 14 + 2F-85 | 21 | 1.49 mm | 5.51 mm |
| RouteBot | UR10e + 2F-85 | 28 | 1.49 mm | 5.90 mm |
| BranchBot | 双 UR10e + 双 2F-85 | 56 | 1.50 mm | 2.54 mm |

视觉配置与声明边界见 `configs/sim/realistic_visuals.json`，机器可读结果见
`artifacts/mujoco_validation/mujoco_manifest.json`。

## Paper 1 — InspectBot

冻结主张：**在真实公开线缆缺陷分数驱动的配对混合基准中，正确绑定的工程拓扑风险图可在
欠覆盖扫描预算下提高关键性加权召回；感知失效、全覆盖和不当复扫会消除或反转该优势。**

公开数据已经完成完整性审计。MVTec cable 使用 180 个 normal fit、44 个 normal calibration
和 58 good + 92 anomaly 的原始 test；Spatial Gaussian image AUROC 为 92.2%，pixel AUROC
为 95.7%。独立 Stripped Wire 重拟合使用 160/40 normal fit/calibration 和 133 good +
167 anomaly test；最强 spatial NN 为 81.5% AUROC，但严格 normal-only 阈值 recall 仅
41.3%。该结果不与使用 labeled-test threshold 的 PB-IAD F1 直接比较。

四视图 score cache 包含 600 行真实 MVTec test 图像分数。clean 与 dark-blur AUROC 约 92%，
但 occluded-dark recall 仅 1.1%。stress-v1 的 1,000 个 adverse-first paired scenarios 因此
形成保留负结果：预算 4--12 接近感知下限，无稳定 topology 优势。

deployment-v2 使用 2,000 个不相交 clean-first paired scenarios。Topology-risk 相对 Geometry
在预算 4/6/8/10 的关键性加权召回提高 9.4/7.8/7.7/4.7 个百分点，配对 bootstrap CI 均不跨
零，且在包含全部 30 个 topology-versus-baseline 比较的 Holm 校正后显著。预算 12 全覆盖
打平；预算 16 的退化视图均值复扫使 topology 反而低 2.4 个百分点。shuffled-risk 负控说明
正确位置—风险绑定有作用，但不能证明风险权重由模型学得。

描述性 oracle 的初版贪心错误已透明保留并更正为精确 multiple-choice knapsack。修正前后
72,000 条非 oracle 行在 17 个共享字段完全一致，paired CSV 字节级相同。直接 MuJoCo 审计
包含 20 seeds × 4 policies × 4 budgets = 320 runs，全部原生到位；最大 FOV 误差 0.97 mm，
最大视觉 IK 误差 1.50 mm，且 14 个共享结果字段与 deployment-v2 零差异。独立审计共
787 项全过；英文 7 页研究草稿已完成并逐页视觉质检。

必须完成：

- 在实体 UR5e/工业相机上做真实多视角闭环，不再用确定性 appearance transform 代替视点；
- 增加端子退针、标签错配、反光、细小破皮和完整汽车线束，而不只用横截面与剥线端图像；
- 增加多训练 seed、风险权重敏感性、学习型 stopping/view policy 与强 active-vision 基线；
- 做跨供应商、相机、光照和工厂的前瞻性外部测试，并冻结真实硬件协议；
- 报告真实节拍、标定误差、碰撞/安全、维护和人机复核流程。

终稿状态：`public-data-frozen / paired-active-signal / negative-control-retained /
direct-mujoco-complete / research-draft-complete / camera-hardware-open`。

## Paper 2 — InsertBot

冻结主张：**接触触发的目标更新、姿态修正、卸载与重试能在合成线缆端子插接任务中显著
优于不使用力觉恢复的直线插入和螺旋搜索；但当前证据不能证明其优于强 guarded-admittance
基线，更不能外推为实体产线性能。**

冻结的 2.5-D 主表包含 8 个策略 × 3 个难度 × 200 个共同物理 seed，共 4,800 条 episode。
hard 条件下 ContactBelief 为 90.0% 验证且无损伤成功、4.5% 损伤代理、10.4 N 平均峰值力；
直线插入为 56.5%、37.5%、11.9 N，螺旋搜索为 46.5%、53.5%、15.4 N。ContactBelief
相对两者的配对成功率优势为 33.5 和 43.5 个百分点，21 项 Holm 校正后均显著。去姿态更新
降至 78.5%，校正后显著；去退回状态为 84.0%，方向一致但校正后不显著。

guarded admittance 达到 86.0% 成功、3.5% 损伤和 9.8 N，ContactBelief 的 4 个百分点优势
不显著，且 guarded 的平均峰值力更低、循环时间更长。这一结果被作为防止过度主张的主要
反证保留。

独立 direct MuJoCo 审计包含 5 个策略 × 3 个难度 × 20 个共同 seed，共 300 条 episode。
hard 条件下 ContactBelief 与 guarded 均为 60% 成功、0% 损伤代理；直线插入为 25% 成功、
75% 损伤，去退回为 35% 成功、65% 损伤。由于每单元仅 20 个 seed，校正后的成功率差异
未显著，该结果只作为直接当前观测和原生碰撞力的描述性跨物理审计。

MIT 许可的 LeRobot ALOHA 公开代理数据包含 50 条完整 episode、25,000 帧，已按 episode
冻结为 40/10 训练/测试。岭回归动作 RMSE 为 0.03271，相对状态保持基线降低 74.88%；由于
它是刚体双臂仿真且没有汽车端子、线缆反力、相机或接触力，该结果只审计数据管线与动作接口，
不进入成功率或力安全主张。

公开数据、主表、MuJoCo 报告、场景、统计量与论文资产已通过 153/153 项独立检查。英文 7 页
研究草稿已完成并逐页视觉质检；它明确标注 `DRAFT—NOT FOR SUBMISSION`。

必须完成：

- 冻结至少三种真实连接器/线缆族，测量实际间隙、摩擦、插入力、锁扣和损伤阈值；
- 在实体机械臂、工业相机和校准六轴力/力矩传感器上完成至少 100 次/关键比较的配对实验；
- 将任务空间代理升级为关节/力矩或末端力控制，并使线缆反力完整作用于端子；
- 增加学习型接触策略、多训练 seed、传感延迟/漂移、未见公差与跨连接器泛化；
- 预注册成功、峰值力、循环时间、重试、卡滞、退针、零件损伤和插后拉拔验证；
- 只有通过硬件门槛后，才讨论真实可靠性、使用寿命、替代人工或客户回本。

终稿状态：`contact-protocol-frozen / paired-force-safety-signal / direct-mujoco-complete /
research-draft-complete / hardware-force-calibration-open`。

## Paper 3 — RouteBot

暂定主张：**显式维护“语义线段—有序卡扣”对应关系，可显著提高障碍环境中的线束入扣
成功率与未见布局泛化。**

当前冻结 PBD 主表包含 3 个难度 × 100 个配对 seed × 12 个策略，共 3,600 条闭环 episode。
TopoHarness 为 251/300（83.7%，Wilson 95% CI `[79.1%, 87.4%]`），等参数 Geometry 为
10/300（3.3%，`[1.8%, 6.0%]`），教师为 300/300。三个难度的配对优势依次为 90、83、
68 个百分点，Holm 校正精确 McNemar 检验均 `p <= 2.29e-18`。但 ACT-RelPool、
Semantic-only、去 unary ID/order 特征以及 20%/50% 数据模型均与 TopoHarness 完全相同地得到
251/300；因此当前证据证明的是“可观测工艺关系必要”，不能证明某个 GNN 架构独有优势。

九关系反事实全量表也已冻结：3 个难度 × 100 个物理 seed × 9 个关系 × 5 个策略，
共 13,500 条 episode。TopoHarness 与 ACT-RelPool 均为 2,340/2,700（86.7%），Geometry
为 57/2,700（2.1%），ACT-Pointer 为 268/2,700（9.9%），教师为 2,700/2,700。
27 个关系—难度单元的 TopoHarness 对 Geometry 优势均为 60–100 个百分点，最大 Holm
校正 `p=1.06e-16`；两个关系模型仍有完全相同的成功/失败分区。

公开 Berkeley Cable Routing 的
1,647 条真实机器人轨迹已经按完整 episode 冻结划分；时序岭基线在 247 条留出轨迹上相对
action persistence 降低 11.58% 的平均 episode RMSE，但这只是离线动作审计。
直接 MuJoCo 观测—任务空间动作闭环已完成 20 个配对物理 seed × 9 种关系：
TopoHarness 和教师各 180/180，Geometry 0/180；精确 McNemar Holm 校正
p=2.61e-54。该表已经达到项目预设的跨物理规模门槛，但仍是 mocap 任务空间代理，
不是关节力矩、接触力或真实硬件证据。

2026-08 文献审计发现 WireCraft 已覆盖工业 DLO 的 connector insertion、clip routing 和
channel seating，DLO-Lab 也已提供可微物理基准。因此 RouteBot 不能再以“首个工业线束
仿真基准”为主张，必须转向语义线段—卡扣关系的反事实稳健性，并优先做 WireCraft/DLO-Lab
外部复核。详见 `docs/ROUTEBOT_LITERATURE_AUDIT.md`。

DLO-Lab 官方源码已固定到 commit `c5026a...` 并完成 `wiring_post` 接口审计。README 的
SharePoint 链接对命令行返回 HTTP 401，但浏览器可公开预览和下载；149,168,403-byte
原包已固定为 SHA-256 `acd483e2...e421`，必要目标与纹理均已恢复。包内没有独立资产
许可证，因此只用于本机外部复核、不得随匿名复现包再分发。完整证据与冻结协议见
`docs/ROUTEBOT_DLOLAB_EXTERNAL_AUDIT.md`。

独立 CUDA 环境、官方 reset/step smoke 和四路径匹配评估现已完成。正确有序路径的
官方原生奖励为 0.9748，错误夹具顺序为 0.5715；但直达终点控制也达到 0.9698。
因此该块验证了官方环境可执行、gross wrong-order 可检测，同时也形成一项重要负结果：
`wiring_post` 的几何目标本身不足以区分 RouteBot 工艺语义。它是外部 sanity check，
不是训练策略的外部语义复现。

必须完成：

- 主表 100+ 配对 seed 已冻结；仍需把更强的关系/布局变化移入主任务，避免关系架构和
  数据量消融饱和；
- 增加 ACT 或 Diffusion Policy 等时序基线，并严格匹配观测和示教预算；
- 做去节点身份、去语义边、坐标-only、不同抓持点和课程学习消融；
- 报告全序列成功、单卡扣召回、错序率、碰撞、线缆应变与循环时间；
- 保留已冻结的九关系、20 物理 seed MuJoCo 直接闭环表，并补充力/力矩级或真实硬件复核；
- 在 WireCraft/DLO-Lab 或真实硬件完成带语义成功标签的闭环外部复核；Berkeley 离线审计
  不能替代此项。

终稿状态：`main-table-frozen / strong-relation-signal / external-sanity-complete /
trained-semantic-replication-open`。

## Paper 4 — BranchBot

冻结主张：**当物理证据保持不变时，显式语义通道与类型化动作落地可避免 A/B 分支身份
捷径；该收益不属于某一种消息传递架构。**

正式 PBD 主表现包含 100 物理种子 × 3 难度 × 2 语义指派 × 10 策略，共 6,000 条闭环
episode。TopoHarness 为 480/600（80.0%，Wilson 95% CI `[76.6%, 83.0%]`），Geometry
为 123/600（20.5%，`[17.5%, 23.9%]`），ACT direct 为 432/600（72.0%），
ACT-Pointer 为 478/600（79.7%），ACT-RelPool 为 479/600（79.8%），教师为 478/600。
Geometry 的 123 次成功全部来自 canonical 指派，A/B swap 后为 0/300；TopoHarness 的
canonical/swap 结果一致率为 99.3%。TopoHarness 对 Geometry 的六个难度—指派单元均在
Holm 校正后显著，最大校正 `p=2.56e-3`。未类型化 ACT 在高难度的两个指派中也显著落后。

三个图消融分别为 477/600（去语义边但保留 unary identity）、478/600（去物理边）和
481/600（去 unary ID/order 但保留语义关系）。这些结果与类型化 ACT 一起表明多种语义
通道都足以达到约 80% 的任务可行性上限，不能把增益归因于完整 GNN 独有能力。

直接当前观测 MuJoCo 闭环包含 20 物理种子 × 2 指派 × 3 策略，共 120 条 episode。
TopoHarness 与教师均为 40/40，Geometry 为 13/40；Geometry 的 13 次成功全部 canonical，
swap 后为 0/20。该过程每步从当前 MuJoCo cable/site/mocap 位姿重建图并直接执行双臂
task-space 命令，不回放 PBD 轨迹。

冻结数据包含 420 条原始专家 episode、43,258 个样本，以及成功且按难度—指派平衡的
160/40/80 train/val/test episode（25,740 个样本，整 episode 和物理 seed 隔离）。七个
学习模型使用同一冻结数据。全量测试为 62 passed + 15 subtests；独立结果审计 182 项全过。
英文 4 页研究草稿已完成并逐页视觉质检。

必须完成：

- 将固定两分支扩展到 3--5 分支和完整置换群，增加不同交叉数、遮挡与非对称物性；
- 增加多训练种子，报告训练不确定性；当前 100-seed 配对只覆盖 rollout 不确定性；
- 加入图像分支身份感知及受控遮挡/颜色/标签反事实，避免直接状态语义输入；
- 报告最终分配、交叉数降低、最小间距、双臂冲突、动作长度与成功率；
- 把 task-space mocap MuJoCo 升级为关节控制、碰撞与力/接触约束；
- 完成实体双臂台架反事实实验和安全边界，不用仿真成功率代替产线性能。

终稿状态：`main-table-frozen / semantic-shortcut-identified / direct-mujoco-complete /
multi-branch-vision-hardware-open`。

## 冻结执行顺序

1. RouteBot：标准基线、100-seed 主表、消融、MuJoCo 策略闭环、论文初稿；
2. BranchBot：反事实主表、多分支扩展、双臂闭环、论文初稿；
3. InspectBot：公开数据、主动检测主表、MuJoCo 扫描闭环和研究草稿已完成；转实体相机闭环；
4. InsertBot：仿真接触主表、MuJoCo 审计和研究草稿已完成；转实体力觉插接硬件门槛；
5. 最后统一做引用核验、统计复算、匿名复现、图表排版和四篇交叉一致性审计。

该顺序只是降低返工，不改变“四篇全部达到终稿”的目标。
