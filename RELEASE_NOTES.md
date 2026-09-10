# GitHub 交付说明 · 2026-09-10

以最终实际运行的 `Habitat_Cup_v4 (1)(1).ipynb` 为依据整理。

- 根目录 `Habitat_Cup_v4.ipynb` 用于复跑：保留原实验参数，清除执行输出，统一两个种子的预算说明。
- `cup_baseline/`、三个入口脚本和 `tests/` 共 25 个原有 Python 文件与最终 notebook 内嵌源码逐字节一致，核心算法没有改动。
- 修复可选录像关闭时仍读取 `SEEDS` / `OUTPUT_ROOT` 的问题；录像开启但会话配置缺失时提示先恢复 Cell 1–5。录像算法与脚本未变。
- `results/recorded_run.ipynb` 保留上传文件原始字节；其中的历史错误、截断输出也保留，便于核对。
- 收录已提供的 8 份 CSV、12 份阶段曲线 JSON、两个任务视频及最新版独立 HTML；未补造缺失的逐 episode 数据或 checkpoint。
- 新增 GitHub 首页说明、忽略规则、源码同步工具与 SHA256 清单。更新详细运行说明和验证范围；随 Colab 内嵌包同步这些文档。

`RELEASE_MANIFEST.json` 记录文件摘要及原始来源。修改仓库后，这份清单仅代表本次交付；如修改算法，使用同步工具更新 notebook，并将新结果保存到独立目录。
