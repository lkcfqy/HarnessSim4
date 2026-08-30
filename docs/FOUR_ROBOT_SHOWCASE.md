# HarnessSim4 四机器人统一演示

## 成果

统一演示把四个真实机器人网格 MuJoCo 场景编排为一条 35.875 秒、1920×1080、24 fps 的
H.264 影片：

1. InspectBot：UR5e + 双目检测头，主动扫描与异常定位；
2. InsertBot：KUKA LBR iiwa 14 + Robotiq 2F-85，接触感知端子插接；
3. RouteBot：UR10e + Robotiq 2F-85，有序卡扣布线；
4. BranchBot：双 UR10e + 双 Robotiq 2F-85，分支身份保持、分离与布置；
5. 结尾同步四工位视图，用同一时间轴直接比较四类工艺。

视频包含章节、淡入与交叉转场、工艺特写相机、进度条和清晰的证据边界。交互页面支持
播放、暂停、拖动以及按章节跳转，并保留四个工位的最终帧说明。

## 可复现生成

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-mujoco-demo --frames 72 --output artifacts/mujoco_validation
PYTHONPATH=src .venv-dlolab/bin/python scripts/render_harnesssim4_showcase.py
PYTHONPATH=src .venv-dlolab/bin/python scripts/audit_harnesssim4_showcase.py
```

正式输出：

```text
artifacts/showcase/harnesssim4_four_robot_showcase.mp4
artifacts/showcase/harnesssim4_four_robot_showcase.html
artifacts/showcase/harnesssim4_four_robot_poster.png
artifacts/showcase/harnesssim4_four_robot_storyboard.png
artifacts/showcase/harnesssim4_four_robot_manifest.json
artifacts/showcase/harnesssim4_four_robot_media_audit.json
```

媒体审计从最终 H.264 文件反向解码尺寸、帧率、帧数、时长、编码、六个章节画面和视觉差异，
并核验原始四个 GIF、海报、章节轨和交互页面哈希。目前 24/24 项通过。

## 声明边界

这是 MuJoCo 数字孪生与脚本轨迹的真实感可视化，不是实体机器人视频。机械臂视觉网格与
任务空间运动不能替代关节力矩控制、碰撞安全、相机/力传感器标定、真实线材物性或客户
产线验证。任何论文、路演或客户演示都必须保留 `NOT HARDWARE VALIDATION` 标识。
