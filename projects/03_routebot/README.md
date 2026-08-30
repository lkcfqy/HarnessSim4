# RouteBot

线缆按序入卡扣机器人。配置见 `configs/sim/route.json`，环境实现在
`src/harnessbench/sim/envs/route.py`。

研究入口：抓持点选择、路径控制、障碍规避以及语义线段—卡扣对应。

```powershell
python -m harnessbench sim-demo --tasks route
python -m harnessbench sim-benchmark --tasks route --episodes 20
```

真实可视化：`mujoco/scene.xml` 与运行时拼装的 UR10e + Robotiq 2F-85 视觉孪生，包含
固定根端、三组 U 型卡扣、障碍、末端抓持约束、安全护栏、工厂灯光和双视角相机。机器人
网格碰撞关闭，因此不改变线束物理；边界见 `configs/sim/realistic_visuals.json`。

ACT-style 标准基线与配对主表：

```bash
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-train-act --tasks route
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-paper --episodes 100
PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-route-counterfactual \
  --physical-seeds 10
```
