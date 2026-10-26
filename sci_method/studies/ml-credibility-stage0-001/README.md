# B2可信机器学习：阶段0独立数学与方法学审查

开始2026-10-25，完成整理2026-10-26。参考提交`f92eabac26092bc383bbf1e7df1731f4a93951fa`；独立分支`codex/sci-ml-credibility-stage0`。

## 决策：B，关键数学/创新性问题未解决，暂不训练

本轮不否定所有ML价值，但否定“相同固定审查器因CNN而增强可信性”的归因。优先申请一个更小的**经典统计与候选预算可行性关卡**，尚未批准；不执行旧512场景或六场景，不训练、不跑Geant4，不正式更换主线。

### 已知、未知与失败分别保留

- **数学已证明：**固定对比、同数据同模型、候选无关审查器的结论与CNN/MAP无关。四等深度层的v=(-1,3,-3,1)为FE核的精确零方向；u0与u1协方差同分布。相同观测不能支持区分它们的结构证据。
- **确定性支持：**完整相关hit-error、Fisher rank3/零方向、nuisance与独立校准的不同作用；完整穿越的角度变化仍保留反例。不是Geant4验证。
- **未达原效应预期：**侧边裁切虽打破等式，总covariance相对差仅6.84675e-5，未达事前0.001诊断期望。门槛不放宽、几何不重选，run状态保留失败。裁切Fisher还降为rank2，不能说所有深度已可辨识。
- **文献支持：**CNN选择后推断、任务级边际覆盖与学习零空间补全已有近邻；不能包装为首次。
- **未知/待验证：**一般未知背景下结构对比的有效选择后检验；ML相对强MAP提议/同审查器的发现或成本增益；足够方法新颖性；物理模型迁移及实测授权。

## 阅读顺序与关联资料

| 文件 | 用途/关联证据 |
|---|---|
| [MATHEMATICAL_REVIEW.md](MATHEMATICAL_REVIEW.md) | FE深度矩、完整误差、等分布、Fisher/nuisance、角度与单位 |
| [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md) | A/B/C比较、选择保护、真实ML增量、至多两个假设、定向文献复核 |
| [NEXT_STAGE_PLAN.md](NEXT_STAGE_PLAN.md) | 校准/测试规模分析、母场景隔离、低成本待批计划及训练门槛 |
| [CHECK_SPEC.md](CHECK_SPEC.md) | 计算前声明的固定输入与数值判据，不是新正式实验协议 |
| [TEST_HISTORY.md](TEST_HISTORY.md) | RED与原效应预期失败，保留原断言，不将绿色测试等同正结果 |
| [audit_checks.py](audit_checks.py) / [test_audit_checks.py](test_audit_checks.py) | study-local确定性审查工具与轻量测试，不修改科学算法 |
| [results/run-001/checks.json](results/run-001/checks.json) | 固定矩阵、数值结果和规划有限和；来源快照及哈希在同目录 |
| [VERIFICATION.md](VERIFICATION.md) | 最终实际测试、边界核验和复现说明 |
| [上轮可行性资料](../ml-direction-feasibility-001/README.md) | 本轮输入，原样保留；其中固定审查器收益/512场景建议以本轮限定为准 |

生成脚本关联`results/run-001/evidence.json`，无图像或新论文结果；没有将第三方全文上传。最后更新时间由本页日期及Git历史追踪。数值工具的`kernel_matches_exact`使用**完整深度积分**参考：全穿越应为true；裁切为false是参考不再适用，不是宣称生产积分错误。裁切另有独立材料变化27/676的解析核对。

## 边界与下一步

只新增本study与现有证据索引入口；旧模型、POCA、Geant4、优化器、实验判据、论文和所有历史输出不动。没有把相关子抽样当独立物理重复。ICAR-LR仍未证明优于成熟BIC/LRT。

建议下一步最多2–3个工作日、≤1 CPU小时、无GPU的审查器/预算headroom研究，先交ChatGPT独立审查再由用户批准。此建议不是自动执行指令。若无法证明ML独立增量，保留经典统计/可辨识性成果，停止ML包装。
