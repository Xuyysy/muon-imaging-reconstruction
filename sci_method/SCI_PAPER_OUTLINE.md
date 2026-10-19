# SCI 论文结构草案（原型阶段）

本提纲只固定叙事骨架和证据边界，不提前填写任何成像改善、成功率或性能结论。标题、正式算法名称、主评价指标与投稿期刊均未冻结。

## 暂定问题陈述

在有限路径覆盖、hit-position uncertainty 和低事件数下，观测似然支持哪些局部散射密度对比？是否能用经过 nuisance profiling 与校准的局部信息决定空间细分，避免把先验偏好误认为数据分辨率？

## 章节

1. **Introduction**：缪子散射成像的低通量背景；PoCA作为历史基线而非真实散射点；统计/MAP重建的发展；RHMT v1 与 NAMT v2 的版本关系；明确本文聚焦“局部可辨识参数对比”而非首次使用 raw hits 或 path likelihood。
2. **Related Work**：MLS-EM；Wang–Schultz–Qi 2009 MAP 论文与 GSM 论文（两篇不同工作）；PoCA误差分布；低统计量与自适应网格；RHMT/NAMT 同一 arXiv 工作的版本演变；MINT/μTRec等近期近邻。逐式列明相同组件与差异，不仅按标题/网络名称比较。
3. **Detector Geometry and Observation Model**：四层 plane geometry；hit 到下游 residual 的线性矩阵；共同上游 hit 导致的 detector covariance；完整残差间相关性及误差假设。
4. **Scattering Forward Model**：Fermi–Eyges 路径积分；非负 voxel scattering density；动量、β、小角、路径直线与高斯近似；covariance-likelihood、梯度和数值条件。
5. **Local Identifiability and Resolution Selection**：参数对比的无量纲化；data Fisher、nuisance Schur complement、先验/正则项分离；若进入 split/merge，须给冻结的候选规则、停止条件、复杂度和伪代码。该章目前仅有手工合成原型，不足以定稿。
6. **Experimental Protocol**：Geant4 版本、物理列表、源谱、动量信息、detector errors、路径与事件预算、随机数流、开发/调参/冻结验证/最终未见测试隔离；输入文件哈希与复现脚本；MLS-EM/固定网格似然/粗细网格等强基线。物理实验尚未批准/执行。
7. **Numerical and Synthetic Identifiability Checks**：解析协方差与 Monte Carlo；零噪声/零散射极限；梯度有限差分；覆盖与盲区合成 Fisher 例；失败例和 nuisance profile。可报告本轮结果，但应明确为 numerical prototype，不等于成像性能。
8. **Results**：仅在预先冻结实验后报告拟合、校准、定位/轮廓、低统计量稳健性和不确定性覆盖；保留所有阴性结果；XY投影不得冒充3D结果。
9. **Discussion**：可辨识范围、模型误差、动量—材料混淆、真实探测器误差、路径覆盖、计算代价及失败条件；讨论正则化对推断和观测信息的不同作用。
10. **Conclusion**：仅总结预先定义范围内被证据支持的结论，不外推自然谱或未测试材料/几何。

## 当前证据状态

- **已有代码证据**：`sci_method/src/sci_method/forward_model.py`、`identifiability.py` 及单元测试。
- **已有合成证据**：`sci_method/studies/forward-prototype-001/results/run-003/`，200条手工路径/场景，固定 seed；没有 Geant4 和真值评价。
- **尚缺关键证据**：真实/拟真 hit-error 预算；不同角度与位置的路径覆盖；独立物理随机流；3D分辨率与校准实验；与 MLS-EM、固定网格 likelihood、RHMT/NAMT/MINT 的可复现公平比较；最终未见测试。
- **禁止的现阶段表述**：不得声称提出已验证的新算法、达到某一空间分辨率、优于基线、适用于自然宇宙线或能从正则化后验方差获得充分观测证据。
