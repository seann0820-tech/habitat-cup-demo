# 已有实验记录

本目录保存本次 Colab 实验导出的汇总 CSV、阶段验证 JSON，以及保留原样的 [运行 notebook](recorded_run.ipynb)。这些文件用于复查已有结果；重新运行请使用仓库根目录的 [Habitat_Cup_v4.ipynb](../Habitat_Cup_v4.ipynb)。

## 跨种子汇总 CSV

以下 8 份文件均位于 [aggregate/](aggregate/)。多数采用长表格式：`metric` 指标、`mean` 均值、`std` 种子间标准差、`n_seeds` 种子数，以及 `n_defined_seeds` 有定义值的种子数。空值或未定义指标不应替换为零。

| 文件 | 内容与口径 |
|---|---|
| `aggregate_metrics.csv` | 按模型、任务及 seen/unseen 汇总成功率、步数、碰撞、前向次数、每 episode / 每成功 GFLOPs 与布局差距 |
| `aggregate_paired.csv` | 同一初始条件配对后的成功率、步数及每 episode GFLOPs 差，再按训练种子汇总；方向为元认知减 baseline |
| `aggregate_n85.csv` | 基础及持续学习的达标种子数、删失种子数、达标 episode / 训练环境步数；仅达标者的均值不能当作所有种子的样本效率 |
| `aggregate_continual.csv` | 各阶段适应 AUC、阶段前后成功率、相对冻结初始模型的变化，以及最终能力、BWT 和平均遗忘；用 `table` 区分子表 |
| `aggregate_continual_paired.csv` | 按任务、已完成阶段及评估条件汇总两组的配对成功率、步数和计算量差 |
| `aggregate_costs.csv` | 基础训练、持续学习各阶段、训练总计及已完成评估调用的成本；包含 episode、步数、元网络推理/更新、世界模型更新与总 GFLOPs |
| `aggregate_calibration.csv` | 闭环评估中每个 episode 首个可执行阶段预测的 Brier、ECE、ROC-AUC、高信心失败，以及四种计算操作占比；分任务及 seen/unseen，不能直接当作 A/B/C 操作表 |
| `aggregate_local_monitoring.csv` | 实际执行步骤上的阶段头、碰撞头原始/校准概率质量，以及一步预测误差 MAE/RMSE；`diagnostic` 标识具体量，episode 内预测具有相关性 |

`aggregate_costs.csv` 的各个 scope 有包含关系，不能将全部行相加得到“总成本”。闭环监控、首个阶段预测和共同轨迹监控对应不同观测集合，也不能混为同一诊断。

## 阶段验证曲线

[adaptation_curves/](adaptation_curves/) 共 12 份 JSON，按 `seed0 / seed1`、`baseline / metacognitive` 分类。每组的 `phase0`、`phase1`、`phase2` 分别对应 A 原始环境、B 通道受阻、C 操作环境拥挤。

每个 JSON 是检查点列表，包含阶段内 `train_episodes`、`train_steps`、`by_task` 成功率，以及计算量、计算操作和监控统计。阶段曲线应读取相应任务的 `by_task` 记录；不同阶段的训练计数重新从零开始。检查点评估当前阶段条件，不能据此补画旧条件在阶段内部的遗忘轨迹。

上传时重名文件的对应关系如下；完整文件映射、SHA-256 和识别依据见 [index.json](adaptation_curves/index.json)。

| 原上传文件名的后缀 | 训练种子 | 模型 |
|---|---|---|
| 无括号后缀 | 1 | baseline |
| `(1)` | 1 | metacognitive |
| `(2)` | 0 | baseline |
| `(3)` | 0 | metacognitive |

模型身份根据元决策计数区分，种子与阶段根据两个任务重算的适应 AUC 和已有分种子结果唯一匹配；不是按上传顺序推定。

## 记录范围与复跑

`recorded_run.ipynb` 保留上传文件的代码、说明与执行输出，未为展示而改写。它记录已显示的日志和表格，**并非完整逐 episode 账本**；其中的历史说明可能与实际执行设置不同，配置以执行代码及输出为准。

本包不含模型 checkpoint、完整逐 episode 记录或评估调用账本，不能仅凭汇总表恢复全部动作轨迹、反事实操作收益或中间模型。需要这些数据时，应取得原始实验目录或重新运行，不能从缺失记录补造数值。

原始任务视频保存在 [media/](../media/)：`cup.mp4` 与 `drawer.mp4`。随文件提供的信息未独立确认 seed、agent 或条件，视频作为任务示例使用，不据此给它们添加训练实例标签或估计平均表现。

根目录 notebook 默认 `USE_DRIVE=True`，复跑结果保存至 Google Drive 的 `MyDrive/HabitatHomeTwoTasks/outputs_meta_v2`；保留新运行的 checkpoint、逐 episode 记录、日志与协议配置，可支持进一步机制分析。
