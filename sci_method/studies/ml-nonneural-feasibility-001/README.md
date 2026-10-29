# B2阶段1：非神经可行性关卡

日期：2026-10-29。基线：`b57ba0e8bbe2c930a7b473a34a0d90901708bda1`；
分支：`codex/sci-ml-nonneural-feasibility`。结论：**建议B，暂不训练**。

单ROI有限样本Gaussian检验成立，但候选排序的独立ML价值尚未确立。
强信号下简单MAP/score已接近条件Oracle上界；弱/高噪声下Oracle本身
功效低。经典加权LR比原Q更有功效。学习初值在当前微型问题缺少摊销依据。
这些是有边界的数学与开发证据，不是Geant4、真实装置或SCI确认验证。

## 入口

| 文件 | 内容 |
|---|---|
| [DIAGNOSTIC_PLAN.md](DIAGNOSTIC_PLAN.md) | 运行前登记的场景、参数、种子、判据、预算；提交10ef793 |
| [STATISTICAL_TEST_REVIEW.md](STATISTICAL_TEST_REVIEW.md) | PSD白化/随机支配证明、精确与保守含义、经典LR对照及功效 |
| [CANDIDATE_BUDGET_REVIEW.md](CANDIDATE_BUDGET_REVIEW.md) | 固定/MAP/score/Fisher/Oracle、选择保护、headroom与失败条件 |
| [SOLVER_COST_REVIEW.md](SOLVER_COST_REVIEW.md) | 实际MLE/标量MAP实现边界、初值面板、端到端成本与H2 |
| [AUTONOMOUS_EXPLORATION.md](AUTONOMOUS_EXPLORATION.md) | 反对意见和至多三个替代方向；数学/文献/未知严格区分 |
| [NEXT_DECISION.md](NEXT_DECISION.md) | 十项直接回答、B建议及终止条件 |
| [VERIFICATION.md](VERIFICATION.md) | 测试、失败、CPU、来源与复现 |

## 实际执行

- 预先登记6个手工母场景×128重复，每次156个等价Gaussian事件；使用精确
  模式充分统计量抽样，保存24576个能量值，未生成大型逐事件训练集。
- 候选统计代码固定提交`82ae010`；唯一数值研究run为
  [results/run-001/](results/run-001/)。源码快照、配置、种子键、原始能量、
  全6144个方法/重复记录、汇总、数值与求解成本在该目录。
- 初始17测试通过；输出审查追加3测试，最终20项通过。另重放1536个种子
  部件、核对6144行选择及统计、检查18份归档文件，见
  [archive-review-002.json](verification/archive-review-002.json)。
- 保留初始缺模块RED及第一次归档核查失败；后者误要求后加测试与运行时
  测试相同，修正为核对实际提交/快照并显式登记新增测试。无数据/统计代码
  修复、无补种子、无改阈值、无重跑科学run。

未运行训练、GPU、Geant4、旧六场景、512母场景方案或完整旧Monte Carlo套件。
既有算法、论文、所有历史结果、paper_notes与主工作区未改；不合并main。
本轮结束停止，待独立审查。
