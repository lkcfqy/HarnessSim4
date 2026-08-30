# 公开数据与许可证记录

## 当前实际使用的数据

| 数据/模型 | 用途 | 规模 | 许可证 | 本项目默认行为 |
|---|---|---:|---|---|
| [Berkeley Cable Routing / LeRobot conversion](https://huggingface.co/datasets/lerobot/berkeley_cable_routing) | RouteBot 真实机器人多视角数据接口、完整 episode 留出、离线时序动作基线 | 1,647 episodes / 42,328 frames / 4 views | 原项目 CC BY 4.0；转换仓库标注 Apache-2.0 | 固定 commit `20a7774...`，下载约 322 MB 全部四路视频并逐文件校验 SHA-256 |
| [LeRobot ALOHA simulated insertion demonstrations](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | 插入示教的数据管线、行为克隆与轨迹级评估代理 | 50 episodes / 25,000 frames / 50 Hz | MIT | 下载约 3 MB 状态动作与元数据 |
| [ALOHA / ACT paper and project](https://tonyzhaozh.github.io/aloha/) | 数据产生方法和时序动作分块参考 | 论文与代码入口 | 以各自仓库声明为准 | 仅引用，不复制代码 |
| [Pretrained ACT checkpoint](https://huggingface.co/lerobot/act_aloha_sim_insertion_human) | 后续视觉时序策略的对照组 | 约 51.7M 参数 | Apache-2.0（模型页声明） | v0 不下载 |

下载命令会生成 `harnessbench_manifest.json`，其中保存每个文件的 URL 来源信息、字节数
和 SHA-256。它还从原始 Parquet 生成可移植的 NumPy 缓存；标准路径使用 PyArrow，
受应用控制限制时可使用纯 JavaScript `hyparquet`，Hugging Face 官方 Dataset Viewer
rows API 是最后兜底。这样即使上游数据将来更新，实验也能发现变化，同时避免部分
Windows 工控环境拦截第三方本机 DLL 后项目无法启动。

RouteBot 的 Berkeley 下载器使用 `sim-download-route-public`，同时生成独立数据卡、
便携 NumPy 缓存和不可变版本清单。当前冻结的公开数据实验把完整 episode 按
70/15/15 划分为 1,153/247/247 条轨迹，绝不把相邻帧随机拆进不同集合。公开数据没有
RouteBot 的“线段—卡扣语义映射”干预或统一闭环成功标签，因此只报告离线动作预测，
不能替代仿真的语义反事实实验，更不能被解释为客户现场成功率。

## 为什么它只能叫“代理数据”

ALOHA 任务是仿真的刚体插入，和真实汽车端子存在四个关键差异：

- 线缆会施加不可忽略的回弹、拖曳和扭矩；
- 端子/护套间隙、倒刺和锁止结构产生复杂接触；
- 反光金属、黑色塑料、遮挡和型号差异增加视觉难度；
- 真实设备需要考虑供料、节拍、磨损、误插损伤和安全认证。

因此 v0 的离线 RMSE 只证明代码和数据接口可复现，不证明真机成功率，也不能用于客户
报价或安全声明。

## 候选的线缆专项公开资源

以下资源适合下一阶段调研，但在完成许可证、任务适配和下载完整性审核前，不会自动
混入训练集：

- [DLO3DS](https://github.com/lar-unibo/DLO3DS)：电缆/软管等柔性线状物体的三维形状
  估计与跟踪代码；
- [MovingCables](https://github.com/holesond/movingcables)：运动电缆分割数据与评估代码；
- [Cable Routing and Assembly using Tactile-driven Motion Primitives](https://helennn.github.io/cable-manip/)：
  视觉与触觉驱动的寻线、布线和插入研究；
- [NIST Assembly Task Boards](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-systems-smart-manufacturing-program/assembly-task)：
  工业装配任务定义和评测板参考。

## 数据治理规则

1. 每个数据源必须记录原始 URL、版本/提交、许可证、下载时间和哈希；
2. 切分单位必须是 episode、采集日、连接器批次或工位，禁止随机拆散相邻帧；
3. 公开数据与德丰/客户数据物理隔离，客户数据不得因论文或开源默认外发；
4. 合成数据、公开数据、实验室自采数据和客户现场数据在报告中分别列指标；
5. 未标注许可证的数据只可列为阅读线索，不得默认训练或重新分发。
