# 缪子散射成像论文 LaTeX 草稿

这是与 `muon_analysis` 实验结果同步维护的工作稿，不预设投稿期刊，也不覆盖原有 Word 文件。

## 在 VS Code 中编辑

1. 用 VS Code 打开整个 `/Users/xuyaoyang/geant4-test` 文件夹。
2. 在资源管理器中打开 `B1/paper/main.tex`。
3. 安装好的 LaTeX Workshop 插件会识别论文；保存后可点击右上角的 **Build LaTeX project**，再点击 **View LaTeX PDF** 预览。`main.tex` 首行指定了 XeLaTeX；本目录 `.latexmkrc` 也将 `latexmk` 固定为 XeLaTeX。
4. 如果 VS Code 刚装好插件却没找到编译器，完全退出并重新打开 VS Code；终端里用 `xelatex --version` 检查。

也可以在终端运行：

```bash
cd /Users/xuyaoyang/geant4-test/B1/paper
latexmk -xelatex main.tex
```

生成的 PDF 在 `B1/paper/build/main.pdf`。`build/` 是中间输出，不需要提交 GitHub。

## 写作和复现规则

- 正文的数字来自 `muon_analysis/*_results/*summary.csv`；不要仅凭图片修改数字。
- `figures/` 保存论文实际使用的图像副本。每次重新跑实验并决定更新论文时，需同步替换对应图、核对图注和表格数值。
- 现在只把理想源下已计算并复核的仿真实验写成结果；五个 U 型场景中的重复扰动事件质量规则属于补充消融，不等于真实探测器验证，也不是已完成的机器学习方法。自然宇宙线谱、真实探测器和机器学习事件筛选仍是后续工作。
- `paired_event_quality_results/paired_*.csv`、`event_quality_localizer_ablation_results/localizer_ablation_*.csv` 与 `event_quality_failure_diagnosis/diagnostic_cases.csv` 是新增结果的数值来源；论文 `figures/` 中三张对应 PNG 是用于编译的副本。失败案例为事后诊断，不能写作独立验证。
- L 型留出验证的数值分别来自 `muon_analysis/heldout_l_shape_results/heldout_l_shape_summary.csv` 和 `heldout_l_shape_paired_summary.csv`；论文使用的两张 L 型图保存在 `figures/`。该测试只更换目标形状，未改变铝材料、探测器和理想入射源；如果据此重新设计规则，同一 L 型数据便不能再次充当新规则的留出测试集。
- 双分离铝块验证的数值来自 `muon_analysis/double_block_validation_results/summary.csv`、`paired_summary.csv`；论文中的两张双目标图是该结果目录 PNG 的副本。局部收缩和保守回退规则在双铝块数据生成前已固定，因此本场景用于检验这些规则，但不能在看过结果后重新调参再称其为独立测试。双目标的单窗口误差不作为主评价量。
- T 型独立检验的数值来自 `muon_analysis/t_shape_independent_results/` 中的结果与判据文件；`figures/t_shape_metrics_5000.png` 和 `figures/t_shape_reconstruction_5000.png` 是论文用图。分半置信度规则及判据在检验前冻结；T 型结果为阴性，之后不得用 T 型调参并仍称为独立检验。
- 失效门控诊断只使用 U、L、双铝块开发数据，来源为 `muon_analysis/split_confidence_diagnosis_results/`；`figures/split_confidence_gate_diagnosis_5000.png` 为论文用图。诊断有助于解释稀疏事件支撑，但不是因果证明，也不是 T 型数据上的二次寻优。
- 分半计数门槛与细节收缩消融的运行前设计见 `muon_analysis/SPLIT_GATE_ABLATION_PLAN.md`，完整配对和汇总结果见 `muon_analysis/split_gate_ablation_results/`；`figures/split_gate_ablation_5000.png` 为论文用图。四组均只在 U/L/双铝块开发集上比较，没有稳定超过粗网格；不得把局部正值写成 T 型独立验证或已确认的新方法。
- 图表中的“输入事件数”指从完整四层击中 CSV 抽取的条数；实际落入成像体积的 POCA 数量更少，二者不可混称。
- 等面积 Top-k IoU 使用真值面积设定预测像素数量，仅用于仿真评价；实际未知目标不具备这个真值条件。
- 文末的待完成清单是写作提醒，不属于可投稿正文。定稿时要补齐并移除。
