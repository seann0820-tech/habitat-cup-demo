# 验证范围与复现实验

验证分为源码/统计检查、带明确测试替身的控制器集成检查，以及原生 Habitat 仿真。前两类通过不代表 agent 已在真实 Habitat 布局中达到某个成功率。交付物不包含测试替身生成的实验成绩。

## 本次交付的实际验证状态

本包依据 2026-09-10 提供的最终 Colab notebook 整理。原始运行记录保存在 `results/recorded_run.ipynb`，其中 Cell 4 的保存输出显示完整测试集 **52 项通过**；其余输出和随包 CSV/JSON 是已有仿真实验的记录。

本次打包环境重新运行 **46 项不依赖 Habitat/PyTorch 的测试，全部通过**（4.614 秒，0 失败、0 错误）。包括真实 NumPy 元网络优化，以及实际控制与实验入口连接明确测试替身的集成检查。日志见 `results/packaging_checks.txt`。本地没有 PyTorch/Habitat，未重新运行 `test_core.py` 的 6 项测试或原生仿真、录像；不把历史 Colab 验证说成本次重新运行。

本次另检查了 Python 3.9 语法、notebook 源码同步、可选录像的会话前置检查和压缩包完整性。25 个原有 `.py` 文件（含 9 个测试模块）与最终运行 notebook 的内嵌版本逐字节一致。根目录 notebook 清除了输出，保留 `[0, 1]`、`pilot` 和 Drive 默认设置；仅更新说明与可选录像的启动检查。原始运行 notebook 保留原样。

## 不依赖 Habitat 的检查

检查元网络真实优化、训练/校准数据隔离、快照恢复、动作价值标签、控制预算、已执行动作的一步预测计费，以及评估冻结。检查 N85、持续学习矩阵、配对 episode 身份、跨种子汇总与未达标值处理。图表检查使用明确的临时测试输入，不作为 benchmark 结果。

在缺少 PyTorch/Habitat 的环境中，部分控制器测试通过实际源码 AST 加载 MPC，连接明确的解析动力学/环境替身。它们验证控制流程、计数和状态管理；不能验证 PyTorch 的实际学习效果、Habitat 接触/渲染或所有场景可达性。

源码和 notebook 代码使用 Python 3.9 语法检查。生成器检查最后一个代码单元为可选录像、代码单元共 9 个、notebook 不含执行输出，以及压缩包完整性。源码包提供 `RELEASE_MANIFEST.json` 中的 SHA256 清单，可与 notebook 内嵌源码比对。

## 在完整 Habitat 环境中运行

在源码根目录执行：

```bash
python -m unittest discover -s tests -v
python -m cup_baseline.run smoke --data /path/to/data --out outputs/environment_check
```

Colab Cell 4 自动执行这些检查。完整依赖下还应覆盖真实 PyTorch 世界模型拟合和 checkpoint 恢复。参考控制器检查只验证固定场景/seed 下的接口与任务，不能替代正式训练/测试。

Cell 7 运行持续学习几何预检，覆盖实验协议的不同初始条件。几何预检使用真实 Habitat，失败需保存 `generation_failure.json` 和完整日志；通过不证明全局操作可达。

Cell 6–8 产生真实训练、持续学习、共同轨迹监测诊断及跨种子结果。每个 seed 必须独立初始化两组、使用匹配几何并保持验证/测试冻结。实验尚未完成或某组缺失时，汇总应明确停止，不能填充推测成绩。

## 不应从本地检查推断的结论

- 元认知一定比 baseline 更准确或更省电；
- 一次 smoke 成功意味着所有布局和障碍实例可完成；
- 几何生成通过意味着机械臂具有全局无碰撞路径；
- 小型网络的预测误差等于文献中的人类元认知噪声；
- 两个训练 seed 的 pilot 足以给出稳定的研究结论。

正式结果必须报告实际种子数、episode 数、缺失/未达标情况、协议版本与 NN 计算估计边界。测试 episode 的置信区间和训练 seed 之间的变异需分开解释。
