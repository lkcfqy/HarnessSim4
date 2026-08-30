# RouteBot 公开真实机器人数据实验

## 冻结数据

- 原始项目：Berkeley *Multi-Stage Cable Routing Through Hierarchical
  Imitation Learning*；
- LeRobot 数据集：`lerobot/berkeley_cable_routing`；
- 固定版本：`20a7774a714a5daf56b220e01635425f2cb9745b`；
- 规模：1,647 个完整 episode、42,328 帧、8 维状态、7 维动作、四路
  128×128 相机；
- 领域：Franka Panda 真实机器人遥操作线缆布线；
- 许可：原项目页面为 CC BY 4.0，LeRobot 转换仓库标注 Apache-2.0；本项目
  同时保留两项记录并遵守更严格的署名要求。

状态动作、episode 元数据和四路视频均已逐文件核验 SHA-256。完整来源、字节数、
哈希和下载时间见
`data/public/berkeley_cable_routing/harnessbench_manifest.json`，数据边界见同目录
`DATACARD.md`。

## 无泄漏协议

固定随机种子 `202613`，只按完整 episode 划分：

| 集合 | Episodes | Frames | 用途 |
|---|---:|---:|---|
| Train | 1,153 | 29,126 | 拟合 |
| Validation | 247 | 6,552 | 只选择岭参数 |
| Test | 247 | 6,650 | 一次性冻结报告 |

测试动作中只有索引 `[0, 1, 2, 5]` 随样本变化，因此主指标只聚合这四个有效
Cartesian 动作维度；全 7 维结果仍保存在 JSON。置信区间以完整测试 episode 为
重采样单位，禁止逐帧 bootstrap。

## 冻结结果

| 方法 | Frame RMSE ↓ | NRMSE ↓ | Episode-balanced RMSE [95% CI] |
|---|---:|---:|---:|
| Zero | 0.2511 | 1.044 | 0.2470 [0.2401, 0.2538] |
| Train mean | 0.2418 | 0.997 | 0.2361 [0.2290, 0.2431] |
| Action persistence | 0.1719 | 0.755 | 0.1711 [0.1654, 0.1769] |
| State ridge | 0.1891 | 0.823 | 0.1841 [0.1795, 0.1893] |
| State + action-history ridge | **0.1536** | **0.668** | **0.1507 [0.1459, 0.1555]** |

相对 action persistence，状态加上一时刻已执行动作的时序岭模型：

- 平均每 episode RMSE 降低 11.58%；
- 配对差值为 `-0.01911`，10,000 次完整 episode bootstrap 的 95% CI 为
  `[-0.02061, -0.01759]`；
- 在 247 个测试 episode 中胜 233、负 14、平 0；
- 双侧精确 sign test `p = 2.33e-52`。

这说明公开真实数据中存在可复现的强时序信号，也说明只看当前状态的静态模型不足。
下一步视觉时序基线必须至少超过 action persistence 和这个可审计的线性时序模型。

## 不能声称什么

公开数据没有 RouteBot 的线段—卡扣语义关系干预，也没有与本项目一致的闭环语义成功
标签。因此上述数字：

- 不是机器人布线成功率；
- 不能验证 TopoHarness 的语义因果主张；
- 不能替代真实硬件或第三方仿真器的闭环复现；
- 不能用于客户报价、安全认证或节拍承诺。

## 复现

```bash
PYTHONPATH=src python -m harnessbench sim-download-route-public --include-videos
PYTHONPATH=src python -m harnessbench sim-inspect-route-public
PYTHONPATH=src python -m harnessbench sim-eval-route-public
PYTHONPATH=src python -m harnessbench sim-render-route-public --timestamp 120
```

冻结输出位于 `artifacts/papers/routebot/public_real/`：模型、留出预测、完整报告、
划分清单、LaTeX 表格、四视角真实数据图和图像来源清单均可独立审计。
