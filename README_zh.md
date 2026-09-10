# Habitat 居家双任务：baseline 与元认知 MPC

这个项目提供一个可在 Google Colab 运行的具身智能实验。Fetch 机器人在 ReplicaCAD 居家场景中分别完成取杯、开抽屉。每个 episode 只执行一个任务，导航是共同的前置过程。

比较两组 agent：

| 组别 | 世界模型与底层能力 | 计算调度 |
|---|---|---|
| `baseline` | 三成员动力学集成、CEM-MPC、同一动作/目标/几何引导 | 固定搜索预算 |
| `metacognitive` | 同样的世界模型结构、动作与任务能力 | 根据预测误差、搜索与执行证据分配搜索；可回退固定 MPC |

实验测量样本效率、计算代价、布局泛化，以及 A→B→C 持续学习中的适应、保留与代价。运行代码不内置预训练 checkpoint 或预填成绩；每次运行会生成自己的 CSV、JSON 和 PNG。本仓库另外保存已提供的真实运行记录与结果（`results/`），以及成果展示页面（`docs/index.html`）。

同一 seed 下两组世界模型以相同参数初始化，各自采集轨迹、更新并保存权重。它们的结构和学习规则相同，训练后的权重可能不同。

算法标识为 `meta_comparison_v2`，环境 schema 为 `home_two_tasks_v2`，几何生成协议为 `passage_v2`。这些标识分别约束算法、观测/任务接口和环境分布，不可互相替代。

## 在 Colab 运行

上传 `Habitat_Cup_v4.ipynb`，按顺序执行 Cell 1–8；Cell 9 录像可选。

| Cell | 操作 | 目的 |
|---|---|---|
| 1 | 展开内嵌源码、设置路径 | 单独上传 notebook 即可运行 |
| 2 | 安装/验证独立 Python 3.9 Habitat 环境 | 不更换 Colab notebook 内核 |
| 3 | 准备 ReplicaCAD/Fetch 数据、定义子进程入口 | 保留完整命令日志 |
| 4 | 回归测试、先验参考 MPC 的六种任务/条件检查 | 在正式训练前检查环境与接口 |
| 5 | 集中设置 preset、`SEEDS`、预算和可选 Drive | 默认一次 pilot 使用 `[0, 1]` |
| 6 | 每个种子分别训练两组模型 | 获得基础 checkpoint 与 seen/unseen 指标 |
| 7 | 必测的几何预检和 A→B→C 持续学习 | 获得适应曲线、保留矩阵及阶段成本 |
| 8 | 共同轨迹诊断、分种子比较、跨种子统计 | 生成 CSV/JSON/PNG，检查表现与监测 |
| 9 | 可选录像 | 展示一个实际执行例子，不计入正式指标 |

无渲染训练可使用 CPU；录像需要可用的 GPU/EGL 渲染环境。选择 GPU 不会自动解决所有图形驱动问题，录像脚本通过实际图像帧检测渲染，并保存失败日志。

### 默认 pilot 预算

以下均为每个 agent、每个训练种子的预算。

- 基础训练 60 个 episode，两任务各 30 个。前 4 个先验采集 episode 包含在这 60 个之内，两组使用相同先验采集。
- 基础验证在 episode 0 和每 10 个训练 episode 执行，每任务 10 个；最终 seen/unseen 每任务各 20 个。共 220 个基础验证/测试 episode。
- 持续学习每阶段 20 个总训练 episode，两任务各 10 个；A/B/C 合计 60 个训练 episode。
- 每阶段每 4 个训练 episode 检查适应；开始和各阶段结束评估 A/B/C。默认合计 840 个持续学习评估 episode。
- 每个种子的持续学习几何预检有 240 个不同初始环境 reset，不执行策略、不学习、不计为训练样本。

默认两个种子、两组共 480 个训练 episode、4240 个基础及持续学习评估 episode；另计参考控制器检查和每种子 30 个共同轨迹监测诊断 episode。监测诊断固定 baseline 的动作，冻结的元监测器只观察，不改变策略、不学习，其成本单独报告。先检查流程时可将 `SEEDS=[0]`，但单 seed 不能检验训练随机性带来的变化。

### 文件位置与恢复

Colab 默认路径：

- 源码：`/content/habitat_cup_demo/code_meta_v2`
- 独立环境：`/content/habitat_cup_demo/habitat-env`
- 数据：`/content/habitat_cup_demo/data`
- 日志：`/content/habitat_cup_demo/logs_meta_v2`
- 结果（默认 `USE_DRIVE=True`）：`/content/drive/MyDrive/HabitatHomeTwoTasks/outputs_meta_v2`

每个种子结果目录例如 `meta_v2_learned_pilot_seed0`，包含 `baseline`、`metacognitive`、`continual_baseline_passage_v2`、`continual_metacognitive_passage_v2` 和 `comparison`。跨种子汇总另存为 `aggregate_learned_pilot`。

Cell 5 默认挂载自己的 Google Drive 保存结果与 checkpoint；若设置 `USE_DRIVE=False`，则保存到 `/content/habitat_cup_demo/outputs_meta_v2`。环境与场景数据仍使用运行时磁盘，重启后需要重新执行 Cell 1–5 恢复环境、路径与配置。

`RESUME=True` 只恢复本版对应 seed/agent 的 checkpoint；世界模型、元模型、校准器和训练缓冲区一起恢复。不同 agent 或算法版本不可交叉续训。基础训练重复执行时可能重新运行最终测试，其计算记录在评估账本。已完整完成且协议/checkpoint 匹配的持续学习结果可复用；中断的持续学习从该 agent 的基础 checkpoint 重新运行，不提供阶段内恢复。

## 模型与文献依据

世界模型预测“这个状态执行这个动作后会怎样变化”。MPC 用它评估有限时域内的候选动作序列，只执行第一个动作，再根据真实新状态重新规划。

元认知增加三个可分开检验的环节：

1. **读出**：从近期预测/执行误差与搜索证据提取监测分数。正常绕障不能仅因暂时远离最终目标而被判为停滞。
2. **概率校准**：用训练内部保留的数据调整阶段成功与碰撞概率；正式验证/测试不用于校准。
3. **控制**：学习执行、追加搜索、重启搜索或回退固定 MPC 的行动价值，实际追加计算全部计费。

这种划分借鉴元认知读出/信心转换模型、对自身决策的二阶推断，以及资源受限的 metacontrol。它是工程设计，不是人类 ReMeta 模型或 active inference 的复现。详细结构、标签及局限见 [METACOGNITION_zh.md](METACOGNITION_zh.md)。环境几何和辅助操作的边界见 [ENVIRONMENT_zh.md](ENVIRONMENT_zh.md)。

两组的世界模型学习次数、批量大小与 4096 条 transition FIFO replay 保持相同。FIFO 表示缓冲区满后先淘汰最早写入的经验。元模型另外保存有界的监督/校准记录并更新；其内存和 FLOPs 单独报告。当前不调节世界模型学习频率、replay 分配或视觉信息获取。

## 如何阅读指标

| 指标 | 定义 | 常见误读 |
|---|---|---|
| N85 | 首次验证成功率达到 85%，且下一检查点确认；单位为累计总训练 episode | `>预算` 是未观测到达标，不能当零或直接按预算求均值 |
| NN FLOPs/episode | 世界模型和元网络的前向计算估计 | 不是焦耳，不含仿真/IK/几何等全部成本 |
| NN FLOPs/success | 所有尝试的计算总和 ÷ 成功次数 | 包含失败成本，不是仅成功 episode 的平均值 |
| 前向次数 | 分批量调用次数与样本前向次数 | 不同 batch 大小下，次数不能直接当 FLOPs |
| generalization gap | seen 成功率 − unseen 成功率 | 负值表示这批 unseen 更容易完成；零差距不自动代表高泛化 |
| 适应 AUC | 阶段验证成功率曲线的归一化面积 | 受初始水平影响，不单独等于学习速度 |
| 保留矩阵 | 行为学完阶段，列为测试条件 | 沿一列往下看旧能力变化 |
| BWT | 最终旧条件表现相对刚学完该条件时的平均变化 | 正值可以与较低绝对能力同时存在 |
| 平均遗忘 | 旧条件学会后的最高记录到最终表现的平均下降 | 零遗忘不代表新环境已经学好 |

持续学习表的“冻结初始 SR”来自整套 A→B→C 开始前的基础模型，不是每个阶段开始前的模型。要估计 B 阶段本身带来的变化，比较保留矩阵 A 行/B 列与 B 行/B 列；C 阶段类推。

验证/测试冻结世界模型、元模型、校准参数、训练记录及训练随机状态；episode 内的误差累积与推理控制仍可变化。禁止利用正式测试成败更新模型或调整成本权重。

单 seed 配对 bootstrap 重采样相同初始条件下的两组 episode 对，区间只对当前训练结果成立。跨 seed 表按独立训练种子汇总，不将所有测试 episode 混作独立模型重复。完整运行日志、逐 episode manifest 和评估账本用于审计计数与环境配对。

## 独立源码与 GitHub

源码 ZIP 用于版本管理、阅读和命令行运行；notebook 的 Cell 1 已包含相同源码，Colab 用户不必额外解压 ZIP。GitHub 可以托管代码，仿真仍在 Colab 或本地 Habitat 环境运行。

在项目根目录、已激活的 Habitat Python 环境中，可使用以下命令运行单个 seed：

```bash
python -m cup_baseline.run train --data /path/to/data --out outputs/seed0/baseline --agent baseline --preset pilot --seed 0
python -m cup_baseline.run train --data /path/to/data --out outputs/seed0/metacognitive --agent metacognitive --preset pilot --seed 0
python -m cup_baseline.run check_domains --data /path/to/data --out outputs/seed0/generation_check --checkpoint outputs/seed0/baseline/checkpoint.pt
python -m cup_baseline.run continual --data /path/to/data --out outputs/seed0/continual_baseline --checkpoint outputs/seed0/baseline/checkpoint.pt
python -m cup_baseline.run continual --data /path/to/data --out outputs/seed0/continual_metacognitive --checkpoint outputs/seed0/metacognitive/checkpoint.pt
python -m cup_baseline.run shadow --data /path/to/data --out outputs/seed0/shadow --checkpoint outputs/seed0/baseline/checkpoint.pt --monitor-checkpoint outputs/seed0/metacognitive/checkpoint.pt --n 5
python -m cup_baseline.compare --baseline outputs/seed0/baseline --metacognitive outputs/seed0/metacognitive --baseline-cl outputs/seed0/continual_baseline --metacognitive-cl outputs/seed0/continual_metacognitive --out outputs/seed0/comparison
```

为每个 seed 建立独立目录后汇总：

```bash
python -m cup_baseline.aggregate --experiments outputs/seed0 outputs/seed1 --out outputs/aggregate
```

本交付包含源码、测试、文档、notebook、已提供的结果与任务视频；大型场景资产、虚拟环境和 checkpoint 不随包分发。交付验证范围见 [LOCAL_VALIDATION_zh.md](LOCAL_VALIDATION_zh.md)。
