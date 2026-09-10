# 元认知具身智能：Habitat 双任务实验

在 Habitat / ReplicaCAD 中，让 Fetch 机器人完成**取杯**与**开抽屉**，比较 Baseline MPC 与加入元认知模块的 MPC。元认知模块监控自身预测、搜索和执行过程，学习何时执行、扩展搜索、重启搜索或回退，以检验样本效率、计算代价、布局泛化和持续学习。

**2 项任务 · 3 种环境条件 · 6 个布局 · 2 个训练种子**。默认 pilot 共 480 个训练 episode；基础及持续学习评估协议覆盖 4,240 个 episode，另计参考控制器检查和共同轨迹监控诊断。

- [成果展示与研究讨论](docs/index.html)：包含方法、学习曲线、计算选择、结果与任务视频；下载后用浏览器打开。
- [Colab 运行入口](Habitat_Cup_v4.ipynb)：清除输出的单文件 notebook，内嵌完整实验源码。
- [已有实验运行记录](results/recorded_run.ipynb)：保留本次 Colab 运行输出；[汇总 CSV](results/aggregate/) 与[阶段曲线 JSON](results/adaptation_curves/)可直接复查。
- 任务视频：[取杯](media/cup.mp4) · [开抽屉](media/drawer.mp4)。视频是实际执行示例，统计结果来自完整评估。

## 实验做了什么

| 组成 | 实现 |
|---|---|
| 任务 | 导航后取杯并抬高至少 12 cm，或将抽屉实际拉开至少 18 cm；均需保持 5 个控制步 |
| Baseline MPC | 三成员 MLP 动力学集成，预测动作后的状态变化；固定预算 CEM 搜索，只执行最优序列的第一个动作后重新规划 |
| 元认知监控 | 读取模型分歧、搜索改善、预测误差、碰撞、执行进展等过程证据，预测阶段完成、碰撞风险与一步误差 |
| 元认知控制 | 共享网络的 Q 头学习 `EXECUTE / EXPAND / RESTART / FALLBACK` 的实际收益与计算成本 |
| 布局泛化 | 4 个训练布局、2 个未见布局；两个任务分别评估 |
| 持续学习 | 依次适应 A 原始环境、B 通道受阻、C 操作环境拥挤；B/C 分别相对 A 改变 |

两组使用相同任务、动作能力、几何引导、世界模型结构、初始化种子与更新规则，各自采集轨迹并学习。训练后的世界模型权重可能不同，因此主比较衡量整个系统的效果。诊断头与 Q 头共享表示；**当前 Q 头不直接读取校准概率**。验证和测试冻结模型、校准器及训练缓冲区。

详细复现说明见 [README_zh.md](README_zh.md)；设计与文献依据见 [METACOGNITION_zh.md](METACOGNITION_zh.md)，任务和物理边界见 [ENVIRONMENT_zh.md](ENVIRONMENT_zh.md)。

## 已有结果

熟悉布局取杯中，元认知组的成功率由 **85% 提升至 90%**，每成功一次的神经网络计算量由 **3.884 降至 3.116 GFLOPs（−19.8%）**；该成本包含失败尝试。

收益依赖任务和训练种子。对四个“任务 × 布局”条件等权平均，两组成功率均为 **93.1%**；每成功计算量的均值由 **2.954 增至 3.247 GFLOPs**。阶段曲线与操作记录进一步显示不同训练实例形成了不同的搜索偏好，为后续检验计算收益、监控可靠性和学习调控提供依据。

这是两个独立训练种子的描述性结果，不能据此确立四项目标的稳定提升。详细分种子结果、删失处理及研究方向见成果展示。

## 在 Colab 复现

1. 下载本仓库中的 `Habitat_Cup_v4.ipynb`，在 [Google Colab](https://colab.research.google.com/) 上传并打开。单独上传这一份 notebook 即可；Cell 1 会展开内嵌源码。
2. 按顺序执行 Cell 1–4：展开源码、安装独立 Habitat 环境、下载场景资产，并检查任务接口。Habitat 使用独立 Python 环境，不需要替换 Colab 内核。
3. 在 Cell 5 检查实验设置。默认与已有结果一致：

   ```python
   PRESET = 'pilot'
   SEEDS = [0, 1]
   USE_DRIVE = True
   RESUME = True
   PHASE_EPISODES = 20
   ```

   `pilot` 为每组、每种子 60 个基础训练 episode，两任务各 30 个；持续学习另有 A/B/C 各 20 个训练 episode。`smoke` 使用 4 个基础 episode，适合检查流程；`research` 使用 300 个基础 episode，并同时调整其他预算，不能与 pilot 结果直接混合。
4. 执行 Cell 6–8：基础训练、持续学习、共同轨迹诊断及汇总。Cell 9 录像可选；将 `RECORD_VIDEO=True` 后选择种子、模型和条件。无渲染训练可使用 CPU，录像需要可用的 GPU/EGL 渲染环境。

默认结果保存至 Google Drive 的 `MyDrive/HabitatHomeTwoTasks/outputs_meta_v2`。源码、仿真环境与场景资产仍在 Colab 临时磁盘。运行时重启后，先重新执行 Cell 1–5 恢复环境与变量，再继续；`RESUME=True` 只有在相应 checkpoint 仍存在时才可恢复基础训练。中断的持续学习从基础 checkpoint 重新运行，完整且协议匹配的持续学习结果可以复用。

正式测试结果不用于校准、调节计算成本权重或选择模型。若改变预算、种子或协议，请另设结果目录，保留原始运行记录。

## 代码与结果导航

| 文件或目录 | 用途 |
|---|---|
| `Habitat_Cup_v4.ipynb` | 从环境安装到指标汇总的 Colab 入口 |
| `cup_baseline/env.py`、`tasks.py`、`generation.py` | 仿真接口、任务定义与 A/B/C 几何生成 |
| `cup_baseline/model.py` | 动力学集成和固定预算规划 |
| `cup_baseline/metacognition.py`、`meta_planner.py` | 过程监控、监督/校准、Q 控制与可变搜索 |
| `cup_baseline/run.py`、`protocol.py` | 基础训练、冻结评估和持续学习协议 |
| `cup_baseline/compare.py`、`aggregate.py` | 配对检查、单种子比较与跨种子汇总 |
| `bootstrap.py`、`download_assets.py`、`record_video.py` | Habitat 环境、数据资产和任务录像 |
| `tests/` | 代码与协议回归检查；notebook Cell 4 自动运行 |
| `results/aggregate/` | 8 份 CSV：任务指标、配对差、N85、持续学习、成本与监控质量 |
| `results/adaptation_curves/` | 按模型和种子分类的 A/B/C 阶段验证曲线，共 12 份 JSON |
| `results/recorded_run.ipynb` | 已有 Colab 运行输出，供复查；重新运行请使用根目录的入口 notebook |
| `docs/index.html`、`media/` | 成果展示与实际任务视频 |

重新运行后，每个种子的结果目录形如 `meta_v2_learned_pilot_seed0`，包含 `baseline/`、`metacognitive/`、两个 `continual_*_passage_v2/`、`shadow/` 和 `comparison/`；跨种子汇总保存在 `aggregate_learned_pilot/`。

基础训练的 N85 使用“首次达到 85%，且下一检查点确认”的规则，单位为两任务合计的训练 episode；未确认达标的运行保留删失状态。阶段适应 AUC 是验证成功率曲线的归一化面积，需结合起点和曲线形状解释。`aggregate_calibration.csv` 的闭环监控与 `shadow/` 中的共同轨迹诊断对应不同评估分布，应分别阅读。

## 上传 GitHub 与使用范围

将交付包解压后，上传**项目根目录中的文件与子目录**，让 `README.md`、`Habitat_Cup_v4.ipynb`、`cup_baseline/` 等直接位于仓库根目录。保留 `results/` 与 `media/` 的目录关系，即可同时展示源码、实验记录和任务视频。无需上传 ReplicaCAD 数据、Colab 环境或大型 checkpoint。

独立 `.py` 源码用于阅读和版本管理；notebook Cell 1 使用内嵌源码。修改独立源码后，在仓库根目录执行 `python tools/sync_notebook_source.py`，将改动同步进 Colab 入口并清除旧输出。该命令不会修改 `results/recorded_run.ipynb`。

当前控制器使用结构化状态与几何信息，RGB 只用于录像；抓握采用距离门控辅助，控制与碰撞包含明确近似。本实验不测视觉策略、真实力控或跨机器人迁移。GFLOPs 是神经网络计算代理量，未测真实电耗，也不包含仿真、渲染、IK 和全部软件成本。Type-3、共享决策/信心计算、PFC 分工与 EFE 的结合属于后续研究方向，当前原型尚未实现。
