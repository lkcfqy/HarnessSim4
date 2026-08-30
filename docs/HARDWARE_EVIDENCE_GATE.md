# HarnessSim4 实体硬件证据门禁

## 目的

四个项目的 MuJoCo 机器人均是可复核的视觉/算法数字孪生，不是实体实验。本协议把最后一段
路变成可执行流程：先冻结实验方案，再采集逐次试验、标定、来源和媒体证据，最后由脚本生成
`hardware_gate_audit.json`。没有这个文件，论文投稿门禁必定保持关闭。

`PASS` 只表示真实机器人证据完整、成对、可追溯且经过独立复核。它不表示算法一定优于基线，
也不表示已经证明产线可靠性、设备寿命、客户回本或安全认证。可信的负结果同样可以通过证据
门禁并进入论文。

## 三层指标框架

| 层级 | 用途 | 典型指标 |
|---|---|---|
| 主结果 | 判定任务是否真实完成 | 正确检测；锁止并通过拉拔；按序全部入扣；身份正确且解交叉 |
| 诊断指标 | 解释为什么成功或失败 | 视角数、重试、插入深度、入扣数、路径长度、节拍 |
| 安全护栏 | 防止用危险动作换取成功率 | 峰值力/力矩、损伤、碰撞、安全停机、人工介入、最小间距 |

门禁要求至少 100 个完整成对物理条件用于每个预注册主比较，并保存逐试验原始行。脚本报告
Wilson 95% 区间、配对成功率差、精确 McNemar 检验和 Holm 校正；这些统计结果不被用来
“强制得到显著性”，而是防止只挑成功视频。

## 四项冻结模板

| 项目 | 主策略与强比较 | 最低真实覆盖 | 关键标定 | 主结果与安全护栏 |
|---|---|---|---|---|
| InspectBot | topology-risk vs geometry/random | 30 标本、5 缺陷族、2 供应商、2 相机，四种成像条件 | 相机内参、手眼、TCP、照明 | 检测正确；视角/节拍；定位误差、停机与人工介入 |
| InsertBot | contact-belief vs guarded/direct | 30 标本、3 连接器族、2 台架，三种扰动 | 六维力觉、手眼、TCP | 锁止+拉拔；重试/节拍；峰值力矩、损伤、停机 |
| RouteBot | TopoHarness vs ACT-RelPool/geometry | 20 标本、3 布局族、2 线缆 SKU，三种布局条件 | 力觉、手眼、TCP | 顺序正确且全部入扣；节拍；力、应变、碰撞 |
| BranchBot | TopoHarness vs ACT-Pointer/geometry | 20 标本、3–5 分支、6 语义排列、2 台架 | 双臂基座、手眼、左右 TCP | 身份正确且无交叉；路径/节拍；间距、碰撞、停机 |

这些是投稿证据的最低模板，不是产品验收指标。真实设备的量程、允许力、损伤阈值和安全速度
必须在具体连接器、线材、夹具与风险评估确定后另行冻结，不能从仿真数值直接复制。

## 数据包结构

每项实体研究使用独立目录，不应把论文输出目录当作原始数据目录：

```text
data/hardware/<study_id>/
  trials.csv
  provenance.json
  calibration.json
  media_manifest.json
  protocol_snapshot.json
  trial_record_template.json
  calibration/...
  media/...
```

- `trials.csv`：一行一次实体试验；`pair_id` 固定同一物理条件，`policy` 区分控制器。
- `provenance.json`：实验室、设备、软件版本、操作者、独立复核者与方案冻结时间。
- `calibration.json`：标定种类、设备编号、有效期、原始文件路径及 SHA-256。
- `media_manifest.json`：力曲线、原始图像、全景/特写视频与对应试验编号及 SHA-256。
- `protocol_snapshot.json`：冻结时协议的逐字节快照；仓库协议、快照和来源记录哈希必须一致。
- `trial_record_template.json`：与该机器人字段完全一致的单次试验输入模板。

所有路径必须位于该研究目录内；脚本拒绝路径穿越、哈希不一致、事后更改协议、缺失值、重复
试验、非成对比较、标定过期、媒体无法追溯，以及“成功但出现损伤/碰撞/人工介入”等矛盾。

## 建立 InsertBot 第一套真机研究目录

先用状态机生成不会覆盖已有文件的空模板：

```powershell
python scripts/hardware_study.py --robot insertbot init `
  --root data/hardware/insertbot_pilot_v1
```

在开始采集之前：

1. 选定三种真实连接器/线缆族和 30 个可追踪标本；
2. 完成风险评估、急停/限速/限力验证和六维力传感器、手眼、TCP 标定；
3. 填写 `provenance.json`，将 `protocol_frozen_before_collection` 改为 `true`，且冻结时间早于
   首次采集；
4. 不再修改 `configs/hardware/insertbot_gate.json`，如必须修改则升级协议版本并重新开始正式组；
5. 先做单独标记的 pilot，不把 pilot 行混入 100 对正式试验。

冻结来源信息和安全复核：

```powershell
python scripts/hardware_study.py --robot insertbot freeze `
  --root data/hardware/insertbot_formal_v1 `
  --study-id insertbot-formal-v1 --site-id defeng-lab `
  --operator-id operator-01 --reviewer-id reviewer-02 `
  --robot-serial ROBOT-SERIAL --software-revision GIT-OR-RELEASE-ID `
  --safety-review-approved
```

把标定原始文件放入研究目录的 `calibration/` 后，分别登记三种必需标定：

```powershell
python scripts/hardware_study.py --robot insertbot register-calibration `
  --root data/hardware/insertbot_formal_v1 --kind force_torque `
  --device-id FT-SERIAL --artifact calibration/force_torque_report.pdf `
  --performed-at 2026-09-01T08:00:00+08:00 `
  --valid-through 2027-09-01T08:00:00+08:00
```

`hand_eye` 与 `robot_tcp` 用同一命令登记。三项标定齐全、文件哈希正确且未过期后，状态机才
允许开始采集：

```powershell
python scripts/hardware_study.py --robot insertbot start `
  --root data/hardware/insertbot_formal_v1

python scripts/hardware_study.py --robot insertbot add-trial `
  --root data/hardware/insertbot_formal_v1 `
  --record-json data/hardware/insertbot_formal_v1/trial_record_template.json

python scripts/hardware_study.py --robot insertbot register-media `
  --root data/hardware/insertbot_formal_v1 --kind force_trace `
  --artifact media/trial-0001-force.csv --trial-id trial-0001
```

每次登记前都应复制模板并填入新的 `trial_id`、`pair_id`、策略、条件和测量值，不要直接反复
修改同一个已登记试验。状态机拒绝重复 `trial_id`、重复 `(pair_id, policy)`、未知策略、越界
时间、非有限数值，以及与成功定义矛盾的数据。

采集完成后运行：

```powershell
python scripts/hardware_study.py --robot insertbot complete `
  --root data/hardware/insertbot_formal_v1

python scripts/audit_hardware_gate.py `
  --robot insertbot `
  --evidence-root data/hardware/insertbot_formal_v1

python scripts/audit_submission_bundle.py --require-submission-ready
```

硬件审计成功后写入 `artifacts/hardware/insertbot/hardware_gate_audit.json`；总门禁随后重新检查
四篇论文。
其余机器人只需替换 `--robot` 和研究目录，四份机器可读协议位于 `configs/hardware/`。

## 当前状态

截至 2026-08-27，没有任何实体试验数据被放入仓库，也没有任何硬件门禁通过。当前是
“四篇可复现研究稿 + 四套硬件预注册模板”，仍不能改写成“4 篇投稿终稿”。
