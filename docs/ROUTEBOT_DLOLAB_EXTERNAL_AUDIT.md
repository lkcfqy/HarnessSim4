# RouteBot × DLO-Lab 外部闭环复核审计

审计日期：2026-08-27

## 结论

DLO-Lab 的 `wiring_post` 是当前公开代码中与 RouteBot 最接近的独立仿真任务。
官方代码、任务资产、CUDA 环境、reset/step 冒烟测试和四路径匹配评估均已完成，
数值证据通过独立哈希与有限性审计。它可以计为**外部执行与指标敏感性 sanity
check**，但不能计为“训练策略的外部语义复现”：官方奖励只约束最终几何形状，
而且直接把控制点拉到终点也几乎达到有序路径的分数。

## 固定的软件快照

- 官方仓库：<https://github.com/UMass-Embodied-AGI/DLO-Lab>
- 本地路径：`third_party/DLO-Lab`
- commit：`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`
- commit 时间：`2026-07-01T00:57:31Z`
- 源码许可证：Apache License 2.0（仓库根目录 `LICENSE`）
- Python：官方要求 `>=3.10,<3.14`，本机 WSL 为 Python 3.12.3
- GPU：NVIDIA GeForce RTX 3080，10 GB；Windows 与 WSL 均可见
- DLO-Lab README 指定的 Mushroom-RL fork：
  `https://github.com/XJay18/mushroom-rl.git`，固定 commit
  `ec3364740627da945b8bab6e01d8151edb0f83f1`，MIT License

## 固定的官方资产包

- 官方入口：DLO-Lab README 中的 `dlo-lab.zip` SharePoint 链接
- 本地归档：`data/public/dlolab_assets/dlo-lab.zip`
- 文件大小：149,168,403 bytes
- SHA-256：`acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421`
- 归档项：21；全部位于单一顶层目录 `dlo-lab/`
- 必要目标：`target_pos/wiring_post_finalpos.npy`，848 bytes
- 必要纹理：`textures/rope01.png`，85,130 bytes

命令行 HEAD、Range GET 和 `download=1` 请求在重定向后返回 HTTP 401；浏览器页面
却能预览 ZIP 目录并提供官方下载按钮。第一次下载因浏览器控制会话重连而只留下
420,451-byte 截断文件，该残片已明确改名为 `dlo-lab.failed-part.zip`；第二次下载
完成后，归档列表、路径穿越检查与 SHA-256 均通过，才解压到官方要求的
`third_party/DLO-Lab/genesis/assets/dlo-lab/`。

## 与 RouteBot 的任务对应

官方 `experiments/envs/env_wiring_post.py` 定义一台 Franka 操作 30 顶点柔性杆，
使绳索形成穿过两个固定立柱的 S 形路径。它从
`genesis/assets/dlo-lab/target_pos/wiring_post_finalpos.npy` 读取 30 点目标，
并用双向 Chamfer 型距离、中心位置与高度惩罚计算奖励/损失。官方脚本提供 PPO、
SAC、SHAC、SAPO、CMA-ES 和梯度轨迹优化入口，并默认使用 100 个并行环境。

这与 RouteBot 的“关系决定应由哪一段通过哪一夹具”并不等价：DLO-Lab 当前
任务目标是最终几何形状，没有 RouteBot 的线段—夹具语义标签、九关系干预或
错段入扣成功判据。因此，即使资产恢复，也需要一个预先声明的适配协议，分别
报告 DLO-Lab 原生几何回报和新增语义成功率，不能用后者改写官方指标。

## 资产许可边界

官方源码仓库根目录是 Apache License 2.0，但外部 ZIP 内没有 LICENSE、NOTICE、
README 或逐资产来源说明。ZIP 包含四个 HDRI、一个木桌 GLB、两个小网格、三张绳索
纹理和若干 NPY 数据。因此当前处理原则是：

- 仅按官方 README 用于本机研究复现；
- 不把 ZIP 或解压资产放入匿名复现归档或公开代码发布；
- 公开包只提供官方入口、精确哈希、文件清单和获取脚本/说明；
- 若论文展示这些资产渲染的图，投稿前须进一步确认各视觉资产的可发表许可；
- 不依赖外部视觉资产的数值结果仍可报告，但必须注明资产获取方式。

当前剩余的可复现阻塞是软件执行与方法适配，不再是文件缺失。DLO-Lab 使用 GPU
Genesis/ROD，并要求 PyTorch、Pink/QP、Mushroom-RL 等独立依赖；这些依赖安装在
`.venv-dlolab`，不修改已冻结 RouteBot 主环境。依赖一致性检查已通过。WSL 的
CUDA 驱动同时提供 `/usr/lib/wsl/lib/libcuda.so` 与 PyTorch CUDA 设备，但该目录不在
Quadrants 的默认 `dlopen` 搜索路径；统一执行脚本只在检测到该文件时把目录加入
`LD_LIBRARY_PATH`。这不修改 DLO-Lab 源码，也不改变仿真配置。

## 已完成的官方执行结果

官方 `wiring_post` 冒烟测试使用一个环境、一次零动作 step：206 维观测和 30×3
顶点状态在前后均全部有限，初始原生奖励为 0.08655，step 后为 0.08655，且
`absorbing=false`。机器可读报告：
`artifacts/papers/routebot/final/dlolab_external/wiring_post_smoke_report.json`；数组
SHA-256 为 `824cc06e...3bff9`。

随后四条确定性 open-loop 路径在同一个四环境 batch 中运行，控制点、物理、目标和
仿真步数配置一致：

| 路径 | 官方原生奖励 | 有序点 RMSE | 关系角误差 | 解释 |
|---|---:|---:|---:|---|
| 无动作 | 0.0888 | 43.58 cm | 1.573 rad | 负控制 |
| 直达终点 | 0.9698 | 2.73 cm | 0.069 rad | 几何捷径控制 |
| 错误夹具顺序 | 0.5715 | 12.30 cm | 0.785 rad | 反关系控制 |
| 正确有序路径 | 0.9748 | 1.46 cm | 0.073 rad | 关系一致路径 |

全部顶点状态有限。完整报告：
`artifacts/papers/routebot/final/dlolab_external/matched_paths/matched_path_report.json`；
数组 SHA-256 为 `4493abbb...fe0d`。有序路径显著优于错误顺序，说明适配指标能检测
gross order violation；但直达终点与有序路径几乎同分，反过来证明官方几何任务不能
独立识别 RouteBot 的工艺语义。这一负面控制必须保留在论文中。

## 后续训练策略外部复现协议

1. 保存原始 ZIP、SHA-256、下载日期、最终 URL 与资产文件清单；明确记录随包无许可证；
2. 在固定 commit 上运行官方 `wiring_post` reset/step smoke test（已完成）；
3. 冻结原生目标、100 个物理 seed、材料参数与两立柱布局；
4. 仅通过适配器把 RouteBot 观测和任务空间动作接到 Genesis，不修改原生物理；
5. 同时报告官方几何回报、碰撞/数值失败、语义顺序成功和每回合耗时；
6. 与 Geometry、ACT-style、官方脚本教师/优化轨迹共享完全相同的物理 seed；
7. 逐回合保存数据，并报告 Wilson 95% 区间、配对 McNemar 检验和失败类型；
8. 在论文中把适配后的语义指标明确标为 RouteBot extension，而非 DLO-Lab 原生指标。

## 投稿边界

论文可以写“已在固定的官方 DLO-Lab 环境完成执行与 open-loop 指标 sanity check”，
并报告上述正、负控制；不能写“RouteBot 学习策略已在 DLO-Lab 泛化”或“完成外部
语义闭环复现”。Berkeley Cable Routing 的真实数据离线审计、MuJoCo 直接闭环和本次
DLO-Lab sanity check 均保留，但三者都不能替代未来的训练策略或真实硬件语义复核。
