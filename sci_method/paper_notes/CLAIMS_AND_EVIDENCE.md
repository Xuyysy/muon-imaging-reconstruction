# 候选论文主张与证据台账

**同步基线：** [`b57ba0e8bbe2c930a7b473a34a0d90901708bda1`](https://github.com/Xuyysy/muon-imaging-reconstruction/commit/b57ba0e8bbe2c930a7b473a34a0d90901708bda1)
此台账区分模型内证明、确定性数值核验、合成实验与待验证假设。链接固定在审查提交，后续结果应追加新证据，不覆盖历史记录。

## E1 — 四个深度层的模型内不可辨识反例

| 字段 | 记录 |
|---|---|
| 候选主张 | 在阶段 0 明确规定的四层等宽深度划分、已知竖直路径、固定 (p=4\,\mathrm{GeV}/c)、小角 Fermi–Eyges Gaussian 协方差模型内，存在两个非负且不同的材料场 (u_0=(1,1,1,1)) 与 (u_1=(0.9,1.3,0.7,1.1))，对应完全相同的观测协方差与观测分布，但指定深度对比 (c^T u) 不同 0.1。 |
| 证据等级 | **解析数学证明**；独立矩积分/矩阵计算提供确定性数值交叉核验。不是 Geant4 或真实探测器实验。 |
| 适用假设 | 层区间为 ([-100,-50],[-50,0],[0,50],[50,100]\,\mathrm{mm})，目标在两探测器组之间；名义探测器平面 (z=(-400,-300,300,400)\,\mathrm{mm})；固定动量、无能损/吸收、Gaussian 小角散射、已知路径、独立各向同性 Gaussian hit 误差。不能外推为所有几何、材料或真实装置的普遍深度分辨率结论。 |
| 来源 | [MATHEMATICAL_REVIEW.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/ml-credibility-stage0-001/MATHEMATICAL_REVIEW.md)，同一 [审查提交](https://github.com/Xuyysy/muon-imaging-reconstruction/commit/b57ba0e8bbe2c930a7b473a34a0d90901708bda1)。 |
| 负面结果/边界 | 对应协方差核矩阵有零方向 (v=(-1,3,-3,1)^T)。增加同一几何的重复事件只放大非零信息，不会解除精确零方向。横向覆盖、层间测量、裁切路径或更完整物理可改变信息结构，但须重新分析。阶段 0 裁切诊断的总协方差变化仅 (6.84674866339067\times10^{-5})，低于预先设置的 0.001 诊断预期；该失败保留，不能包装为可检出改善。 |
| 尚缺少 | Geant4/真实几何下的可辨识性审查；非直线路径和层依赖动量/能损等物理效应；实际探测器误差预算；多场景验证。 |
| 当前可用表述 | “在所列固定路径、固定动量和 Gaussian Fermi–Eyges 协方差模型假设下，存在精确不可辨识的深度对比反例。”不得省略适用条件，不得写成真实系统普遍无法恢复深度结构。 |

## E2 — 固定、候选无关统计审查器不因 CNN/MAP 前端而改变结论

| 字段 | 记录 |
|---|---|
| 候选主张 | 若 CNN 与 MAP 使用相同观测 (D)、预声明对比 (H)、统计模型/校准 (M)，且统计审查器 (R(D,H,M)) 完全不读取候选图像及其选区，则二者产生完全相同的结构支持、拒判、profile 区间和检验结果。 |
| 证据等级 | **条件成立时的函数恒等/数学论证**，不是性能实验或神经网络比较。 |
| 适用假设 | 输入、预声明对比、统计模型、校准信息、数值精度和审查随机流相同；审查器不以候选图像、图像选区或候选估计改变参数/对比。 |
| 来源 | [METHODOLOGY_REVIEW.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/ml-credibility-stage0-001/METHODOLOGY_REVIEW.md)。 |
| 负面结果/边界 | 因此不能把固定审查器本身的有效性、功效或可靠性提升归功于 CNN。若 CNN 选择候选再检验，会有选择后偏差；独立训练并不能自动消除此偏差。事件拆分、全族同时保护或有效选择后推断均有假设与功效成本。 |
| 尚缺少 | 一般复合协方差结构零假设下的有限样本有效检验；候选选择后误差控制；同 MAP 强基线上的 ML 独立效率/质量增益。 |
| 当前可用表述 | 可用于说明研究设计边界和为什么不能将审查器结果归因于 CNN；不能声称 CNN 已提升可信度、检验功效或错误率。 |

## E3 — 完整测量误差相关协方差、Fisher 与路径信息

| 字段 | 记录 |
|---|---|
| 候选主张 | 在原型四层名义几何和独立 hit (x/y) Gaussian 误差假设下，两个下游残差共享上游 hits，测量残差协方差含非零相关项；路径积分散射协方差也含下游残差间协方差。对预先给定路径与参考点，data Fisher 刻画局部参数/对比信息，须与先验精度分开报告。 |
| 证据等级 | **解析推导 + 确定性代码/单元核验 + 有限合成 run-007**。kick Monte Carlo 是散射积分实现的数值交叉核验，不是 Geant4 验证。 |
| 适用假设 | (z=(-400,-300,300,400)\,\mathrm{mm})，固定 (p=4\,\mathrm{GeV}/c)、(eta\simeq1)、已知名义直线路径、组间小角散射、固定网格分段常数散射密度、独立各向同性 Gaussian hit (x/y) 误差；未包含自然动量谱、能损、吸收、非 Gaussian 尾、磁场、空气/支撑结构及真实读出效应。 |
| 来源 | [FORWARD_MODEL.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/FORWARD_MODEL.md)；run-007 配置、手工路径、合成观测与信息矩阵位于 [`forward-prototype-001/results/run-007/`](https://github.com/Xuyysy/muon-imaging-reconstruction/tree/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/forward-prototype-001/results/run-007/)，证据索引见 [RESEARCH_EVIDENCE_INDEX.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/RESEARCH_EVIDENCE_INDEX.md)。 |
| 可核对数值 | 每投影探测器残差块为 σ²[[86,98],[98,114]]；σ=0.3 mm 时非对角项为 8.82 mm²。run-007 使用 200 条手工路径、固定参考密度与 σ_hit=0/0.3 mm。100,000 条、seed 20261020 的离散 kick 核验最大相对误差 0.238%（3%预设容差）。数值只描述该实现与固定场景。 |
| 负面结果/边界 | 四段深度模型的一个近零方向仍存在；给先验可提高后验曲率，但 data Fisher 不变。局部 Fisher 不是全局唯一恢复保证；奇异矩阵伪逆给出有限数值不能证明目标对比可估计。Fisher 数值秩和 hit 误差影响不能直接外推为真实装置性能。 |
| 尚缺少 | 带噪轨迹估计的 errors-in-variables 审查；真实对准、效率、空间分辨率与材料模型；自然宇宙线动量/角度谱迁移验证。现有 Geant4 source 的 4 GeV 是动能，而前向模型采用 4 GeV/c 动量，不可混为相同条件。 |
| 当前可用表述 | 可报告模型公式、协方差结构与固定合成场景中的数值；必须明确这是理想化前向模型和开发证据，不是自然宇宙线或实测性能结论。 |

## E4 — 历史负面、失败与数值优化审计

| 字段 | 记录 |
|---|---|
| 候选主张 | 现有 ICAR-LR/split-merge 原型揭示了均匀材料误细分、强基线竞争、低功效拒分、代数混淆及优化数值认证等限制；这些结果用于约束后续论文表述，不构成成功算法证明。 |
| 证据等级 | **合成开发实验与数值优化审计**；不属于 Geant4 或独立 SCI 验证。各 run 协议、种子及源码版本不同，须逐 run 引用，不能混为同一估计。 |
| 来源 | [ADAPTIVE_AUDIT_FIX_RESULTS.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/ADAPTIVE_AUDIT_FIX_RESULTS.md)、[adaptive prototype README](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/adaptive-prototype-001/README.md)、[OPTIMIZER_STABILITY_REVIEW.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/OPTIMIZER_STABILITY_REVIEW.md)、[阶段0裁切记录](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/ml-credibility-stage0-001/MATHEMATICAL_REVIEW.md)。 |
| 已知负面 | 原 run-001 的均匀场 adaptive split 为 6/200，95% Wilson 区间 1.38–6.39%；同批旧记录的 BIC 为 0/200，完整规则为 6/200，故没有证明优于 BIC。修正生成器/似然协方差一致性后的新 run-002 均匀场为 16/200；run-001 与 run-002 使用不同随机流，不能作配对因果效应。修正 run-002 的强对比 200/200、无右侧路径 0/200、弱对比/高 hit 误差 0/200、代数混淆负例 0/200。完全等价模型的近零正 LRT 曾有 LRT-only 表观选择；属浮点/优化余量而非可分辨统计证据，不曾为此调整阈值。 |
| 优化审计 | 固定 8-case/16-fit 数值面板中，旧优化器标记 14/16 成功、2/16 停滞；旧“成功”中 11/14 按后续固定 KKT 标准未达标。候选 NumPy 求解器在该有限面板为 16/16 通过审查，最大 KKT 1.34e-8，独立参照最大 NLL 差 3.64e-12，二元细分决定改变 0/8。该结果只说明有限数值面板的优化器审查/修复，不证明统计方法更优、全局收敛或总体速度优势。 |
| 其他边界 | 历史六场景低预算实验未运行。阶段 0 侧裁切对照的总协方差变化未达预定 0.001 诊断预期，属于未达到预期的负面诊断。 |
| 尚缺少 | 预先冻结协议下的独立合成验证、足够场景数的误分率/功效区间、Geant4 及实测迁移、与强统计基线的确认比较。 |
| 当前可用表述 | 仅可如实描述各 run 的合成开发结果和数值审计，并明确限制；不可写成 ICAR-LR 已有稳定性能优势、已验证的低统计量改善或已解决真实探测器问题。 |

## E5 — 尚未证实的机器学习贡献

| 字段 | 记录 |
|---|---|
| 候选主张 | H1 学习候选排序在相同结构审查预算和错误支持风险下增加真实、受数据支持的发现；H2 学习初值在同目标函数、同 KKT/NLL 与同审查精度下减少全流程求解成本。 |
| 证据等级 | **待验证研究假设**。没有本项目正式训练、测试集性能结果或真实数据验证。 |
| 来源 | [B2阶段0 README](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/ml-credibility-stage0-001/README.md)、[NEXT_STAGE_PLAN.md](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/ml-credibility-stage0-001/NEXT_STAGE_PLAN.md)及本轮研究问题登记。 |
| 负面/风险 | 固定候选无关审查器的结果不因 CNN 前端改变；候选由 ML 根据同一图像选择后再检验会有选择后偏差；训练先验不能恢复不可辨识信息。MINT、NAMT/RHMT、Bayesian/MAP、神经场与不确定性文献已构成近邻，不能把“使用神经网络/物理损失/UQ”单独作为新颖性证据。 |
| 尚缺少 | 一般复合结构检验有效性、独立候选选择风险控制、对经典 MAP 候选方法的独立增益、全流程成本核算、按独立母场景隔离的外部测试、Geant4/真实数据适用性，以及更新文献矩阵后的创新性审查。 |
| 当前可用表述 | 可以写作研究问题与待验证假设；不得表述为已实现、已训练、性能提升、可靠性已证明或已通过真实数据验证。2026-10-27 批准的非神经可行性关卡只标记为“已批准、待结果”。 |

## 文献与证据使用提示

近邻论文和版本比较见 [文献可行性矩阵](https://github.com/Xuyysy/muon-imaging-reconstruction/blob/b57ba0e8bbe2c930a7b473a34a0d90901708bda1/sci_method/studies/ml-direction-feasibility-001/LITERATURE_REVIEW.md)。RHMT v1 与 NAMT v2 是同一 arXiv 记录的不同版本，不应当作两篇独立论文。文献矩阵不等于系统综述，也不构成“首次”或创新性成立的证据。
