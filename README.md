# Habitat 中的元认知 MPC

这个实验研究机器人如何决定**还要不要继续规划**。在 Baseline MPC 上加入过程监控与计算选择，让控制器根据预测误差、搜索改善和执行反馈，学习何时执行、继续搜索、重启或回退。

Fetch 在 ReplicaCAD 中完成取杯和开抽屉。实验包含 4 个训练布局、2 个未见布局，以及原始环境、通道受阻、操作环境拥挤三个条件；用两个独立训练种子比较任务表现与计算成本。

[Colab notebook](Habitat_Cup_v4.ipynb) · [实验结果](results/) · [研究报告](docs/index.html) · [取杯视频](media/cup.mp4) · [抽屉视频](media/drawer.mp4)

研究报告为独立 HTML，下载后可在浏览器打开。

## 结果

熟悉布局取杯中，成功率从 85% 提高到 90%，每成功计算量从 3.884 降至 3.116 GFLOPs，减少 19.8%。这里的成本包含失败尝试。

收益尚未在其他条件和两个种子间一致出现。四个“任务 × 布局”条件等权平均，两组成功率均为 93.1%，每成功计算量的均值从 2.954 增至 3.247 GFLOPs。不同种子还形成了不同的搜索偏好：一个偏向重启，另一个偏向扩展。下一步需要用预算匹配对照和操作日志，区分搜索策略、监控信息与世界模型学习各自的影响。

[汇总 CSV](results/aggregate/) 和[分种子阶段曲线](results/adaptation_curves/)保留了这些差异。GFLOPs 估计神经网络计算量，不代表机器人实测电耗。

## 设计取舍

Baseline MPC 使用三成员动力学集成和固定预算 CEM 搜索。元认知版本保留相同的动作、目标和世界模型学习规则，通过一个共享网络输出三个诊断量和四个计算操作的 Q 值。两组从相同参数初始化，随后各自采集数据和学习；因此结果比较的是完整系统，不能把差异全部归因于监控器。

几个实现中刻意保留的约束：

- **监控与控制分开评估。** 阶段完成、碰撞和一步误差预测用于诊断；Q 头决定计算操作。校准概率没有直接接入 Q 头，概率更准确不必然使控制更好。
- **校准按 episode 留出。** 相邻步骤高度相关，整个 episode 一起划入校准集，不参与共享网络和 Q 头训练。正式验证与测试保持参数、训练缓冲区和训练随机状态不变。
- **回退也计费。** 调用 Baseline MPC 前已经花掉的搜索和监控计算仍计入成本；多算是否值得，由真实执行结果检验。

控制器读取结构化状态与几何信息，RGB 用于录像。抓握采用距离门控辅助。任务定义、观测和物理近似见[环境说明](ENVIRONMENT_zh.md)；监督信号、计算选择和文献依据见[元认知模块说明](METACOGNITION_zh.md)。

## 运行

将 [Habitat_Cup_v4.ipynb](Habitat_Cup_v4.ipynb) 上传至 Colab，按顺序执行 Cell 1–8。notebook 内嵌源码，并安装独立的 Habitat 环境。Cell 9 录像可选，需要可用的 GPU/EGL 环境。

Cell 5 的默认设置与已有实验一致：

```python
PRESET = 'pilot'
SEEDS = [0, 1]
USE_DRIVE = True
RESUME = True
PHASE_EPISODES = 20
```

`pilot` 为每组、每种子 60 个基础训练 episode，持续学习另有 A/B/C 各 20 个。两组、两个种子共 480 个训练 episode，基础及持续学习协议含 4,240 个评估 episode。`smoke` 将基础训练设为 4 个 episode；`research` 设为 300 个，并同时改变搜索、更新和评估预算。

结果默认保存到 Drive 的 `MyDrive/HabitatHomeTwoTasks/outputs_meta_v2`。Colab 运行时重启后，先执行 Cell 1–5 恢复环境和配置，再从已有 checkpoint 续训。中断的持续学习会从基础 checkpoint 重新开始。逐单元说明、命令行运行和指标定义见[复现指南](README_zh.md)。

## 阅读与修改代码

建议从这三个位置开始：

1. [`MPC.act`](cup_baseline/model.py)：固定预算规划及动力学模型。
2. [`MetaMPC.act`](cup_baseline/meta_planner.py)：计算选择循环，以及真实转移反馈如何进入下一步监控。
3. [`MetaMonitor`](cup_baseline/metacognition.py)：诊断监督、TD 目标、校准和状态恢复。

训练与冻结评估入口在 [`run.py`](cup_baseline/run.py)，配对比较和跨种子汇总分别在 [`compare.py`](cup_baseline/compare.py) 和 [`aggregate.py`](cup_baseline/aggregate.py)。

在安装完整依赖的环境中运行测试：

```bash
python -m unittest discover -s tests -v
```

测试重点覆盖训练/校准隔离、TD 时间步连接、恢复一致性、评估冻结和计算成本记账。环境依赖及验证范围见[验证说明](LOCAL_VALIDATION_zh.md)。

修改独立源码后，更新 Colab 的内嵌副本：

```bash
python tools/sync_notebook_source.py
```

同步会清除入口 notebook 的旧输出。原始运行记录保存在 [`results/recorded_run.ipynb`](results/recorded_run.ipynb)，不会被同步命令修改；新实验请使用独立结果目录。
