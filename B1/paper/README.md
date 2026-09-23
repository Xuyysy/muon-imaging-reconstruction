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
- 现在只把已验证的理想源实验写成结果；自然宇宙线谱、真实探测器、机器学习事件筛选仍是后续工作。
- 图表中的“输入事件数”指从完整四层击中 CSV 抽取的条数；实际落入成像体积的 POCA 数量更少，二者不可混称。
- 等面积 Top-k IoU 使用真值面积设定预测像素数量，仅用于仿真评价；实际未知目标不具备这个真值条件。
- 文末的待完成清单是写作提醒，不属于可投稿正文。定稿时要补齐并移除。
