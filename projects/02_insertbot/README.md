# InsertBot

端子对准、插壳、接触恢复和锁止验证机器人。项目的核心研究问题是：当视觉估计存在偏差、
线缆产生拖拽、端子姿态不准且接触可能造成卡滞时，控制器应在何时继续插入、何时根据
力/力矩迹象修正目标，以及何时退回、重新对准并重试。

## 当前正式证据

- MIT 许可的 LeRobot ALOHA 公开插入代理数据：50 条完整 episode、25,000 帧，按 episode
  冻结为 40/10 训练/测试；岭回归动作 RMSE 0.03271，相对状态保持基线降低 74.88%，仅作为
  数据管线与动作接口审计，不解释为汽车线束插接成功率；
- 冻结的 2.5-D 成对基准：8 个策略 × 3 个难度 × 200 个共同物理 seed，共 4,800 条 episode；
- hard 条件下 ContactBelief 为 90.0% 成功、4.5% 损伤代理、10.4 N 平均峰值力；
- 相对直线插入提高 33.5 个百分点并降低 1.5 N 峰值力，21 项 Holm 校正后仍显著；
- 最强经典基线 guarded admittance 为 86.0% 成功、3.5% 损伤、9.8 N，因而目前不能声称
  ContactBelief 显著优于强力控基线；
- 独立 direct MuJoCo 审计：5 个策略 × 3 个难度 × 20 个共同 seed，共 300 条 episode；
- MuJoCo hard 条件下 ContactBelief 与 guarded 均为 60% 成功、0% 损伤代理，直线插入为
  25% 成功、75% 损伤代理；样本量仅 20，作为跨物理描述性复核；
- 公开数据、主表和 MuJoCo 的独立结果审计 153/153 项通过；7 页英文研究草稿已生成。

正式协议与结果：

```text
docs/INSERTBOT_CONTACT_PROTOCOL.md
docs/INSERTBOT_MUJOCO_PROTOCOL.md
artifacts/papers/insertbot/formal/contact_v1/
artifacts/papers/insertbot/formal/mujoco_direct_v1/
artifacts/papers/insertbot/formal/audit/insertbot_result_audit.json
papers/02_insertbot/main.tex
output/pdf/02_insertbot_research_draft.pdf
```

## 运行

```powershell
python -m harnessbench sim-demo --tasks insert
python -m harnessbench sim-benchmark --tasks insert --episodes 20
wsl -d Ubuntu-24.04 -- env PYTHONPATH=src .venv-dlolab/bin/python scripts/run_insertbot_formal.py
wsl -d Ubuntu-24.04 -- env MUJOCO_GL=egl PYTHONPATH=src .venv-dlolab/bin/python scripts/run_insertbot_mujoco_direct.py
wsl -d Ubuntu-24.04 -- env PYTHONPATH=src .venv-dlolab/bin/python scripts/audit_insertbot_results.py
```

## 三维场景与证据边界

`mujoco/scene.xml` 与运行时拼装的 KUKA LBR iiwa 14 + Robotiq 2F-85 视觉孪生包含弹性线缆、
可碰撞柔顺端子、四条插座导轨、后挡块、安全护栏、工厂灯光和双视角相机。控制器直接读取
当前 MuJoCo 位姿与原生接触力，不回放 2.5-D 轨迹；但机械臂仍是视觉/任务空间代理，线缆
反力没有完整传入端子，也没有真实连接器锁扣、相机标定、六轴力传感器或实体机器人数据。
因此当前稿件是可复核的仿真研究稿，不是投稿终稿，也不能据此承诺产线成功率、使用寿命、
替代人工数量或客户回本周期。
