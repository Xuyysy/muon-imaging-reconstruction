# 六场景执行入口质量闸门审查

日期：2026-10-21。分支：`codex/sci-low-budget-execution-gate`。
基线：`35e6553b24cb5db880944b3938250fc3aef2fad3`。
实现与测试证据源码：`b1180b5`。本轮只运行软件单元测试。

## 交付范围

保护现有 `run_adaptive_audit_scenarios.py` 的六场景入口；保留原场景、
随机流、事件分配、alpha=0.05、功效0.80、最小差异0.5、比较方法和评价指标。
优化器、联合 Fisher、功效函数、生成器 `_simulate_low_budget` 与 `_counts`
均未修改。另在旧 standalone prototype 的源码快照清单补入新增优化器模块；
本轮没有把旧 prototype 的其它实验入口改造成六场景入口。

新增 study-local 适配器 `studies/low_budget_execution_gate.py`，避免重写
优化器及公共科学模型。详细实施标准先记录于
[LOW_BUDGET_EXECUTION_GATE_PLAN.md](LOW_BUDGET_EXECUTION_GATE_PLAN.md)。

## 1. 校准拟合与失败保护

每个输入在拟合前保存组名、事件分配、随机种子、阶段/重复编号、
`n_events`、完整 `Y.T@Y` 及标准化 float64 观测内容 SHA-256。
在已知零均值高斯模型中，n与散布矩阵足以重算当前似然；它们不等于可恢复
每一条原始残差。源代码、参数和种子用于重生成，内容哈希用于核验。

每次粗/细拟合记录初值、状态、退出原因、参数、NLL、梯度、报告KKT、
重算KKT及内部参数方向的Hessian特征值。数值认证同时要求：

- 状态明确为 converged，迭代数与数值字段有效，参数可行；
- 按原固定坐标尺度重算KKT不超过1e-6，报告KKT也须达标；
- 原始目标实现、白化目标实现及求解器目标值在既定精度预算内一致；
- 报告与重算梯度一致，未发现内部参数方向显著负曲率；
- 预测NLL有限。求解器Boolean或报告的零KKT不能单独通过认证。

`fits.jsonl` 先写完整结果并flush/fsync；未认证结果随后写
`failures.jsonl`，再抛出 `ExecutionGateError`。粗拟合失败时，细拟合明确
记为尚未尝试，不继续消耗后续样本。细拟合失败时保留已经认证的粗拟合。
不存在筛选成功拟合后计算阈值的分支，也不更换失败种子或自动恢复该run。

成功校准行逐行写CSV；如果第2个样本失败，第1个成功行仍可审查，
但整个run为 aborted，不能将这个不完整CSV当作完成的校准总体。

## 2. 原始与稳定LRT

仅支持当前 known_fixed 嵌套关系：
`(parent,other) -> (parent,parent,other)`。使用前逐组验证观测、顺序、固定
背景/测量协方差完全相同，且粗导数等于细导数乘映射矩阵。
背景nuisance参数不同的模型不能直接复用该映射。

保存 `2*(NLL_coarse-NLL_fine)` 的原始有符号值，以及原方案的
`max(0,raw)`。另在细模型内从细终值到嵌入粗终值计算稳定协方差目标差。
稳定值用于数值核查；校准阈值与评价决策继续使用原始统计量。

沿用上轮数值审查每个拟合的预算
`b=max(1e-8,512*eps*max(1,abs(NLL)))`，并计入观测到的目标实现差异。
粗细目标差的诊断容许量为两者之和的2倍。非有限值、显著负LRT或原始/
稳定差异超出该量时，先保存再中止。接近零的分数只标注数值分辨限制，
不会通过事后归零或改阈值改善实验结果。

每个校准层必须保留全部预定拟合。原始LRT按既有alpha/order statistic
计算正式cutoff；稳定LRT以同样规则产生诊断cutoff，记录二者差异。
校准容许量取该层所有样本诊断量的最大值，以反映order statistic扰动。
评价时，校准阈值比较计入“本次拟合量+校准量”；BIC比较计入本次拟合量。
如果任一原始/稳定比较不一致，或其诊断区间接触cutoff，就记录歧义并停止。

这些量是预先声明的保守数值诊断预算，不是严格优化误差上界或统计置信区间。
本轮没有改alpha、功效、最小差异或任何历史阈值。

## 3. 运行生命周期与来源

创建新目录后，在工作开始前保存配置、环境、源码commit、源码快照和SHA。
清单包含 `covariance_optimizer.py`、入口适配器、原生成器/比较代码、
Fisher/forward模块和冻结计划。旧 standalone runner 的新快照也包含优化器。

每个run包含：

| 文件 | 用途 |
|---|---|
| config.json / environment.json | 冻结场景、种子方案、数值规则、版本和NumPy构建信息 |
| inputs.jsonl | 校准、训练和验证输入的充分统计量、计数和内容哈希 |
| fits.jsonl / not_attempted.jsonl | 已尝试拟合的详细认证结果，以及未尝试模型的明确记录 |
| lrt_checks.jsonl / calibration_cutoffs.jsonl | 原始/稳定LRT、映射核验、误差预算、阈值与歧义 |
| low_budget_null_calibration.csv / low_budget_trials.csv | 已完成且通过数值闸门的逐次结果；可为部分文件 |
| failures.jsonl / failure.json | 数值故障及顶层异常、上下文和traceback |
| current_trial.json / run_state.json | 当前样本身份；running、completed或aborted状态 |
| source_snapshot/ / evidence.json | 工作开始前的源码，以及退出后的全部输出哈希 |

JSONL每条记录写盘。Python异常和KeyboardInterrupt经过顶层finally更新状态
与哈希清单。硬杀进程、断电和磁盘故障不能由finally保证；启动清单和已flush
记录可用于辨认不完整run。目录已存在时，任何写入前拒绝执行。

原 `coarse_fit_ms` / `fine_fit_ms` 仍只衡量求解器时间，新增的认证和预测计时
另存 `certification_and_scoring_ms`，没有悄悄改变原时间评价指标。

## 4. 测试结果与失败注入

完整套件 **82项通过：原61项 + 新21项**。归档入口仅调用unittest，
没有运行六场景命令或Geant4。测试使用零数组/手工参数与注入状态，
不能作为缪子成像的实验数据。每个归档夹都有unit_fixture标记。

| 注入或检查 | 结果 |
|---|---|
| 第1个校准粗拟合失败 | 仅尝试该样本；细拟合未尝试；阈值函数0次调用 |
| 第2个校准样本失败 | 第1条成功CSV行保留；种子保持base、base+1；没有子集校准 |
| 细拟合失败 | 粗/细两份结果均保留，随后停止 |
| status声称成功、KKT伪造为0 | 重算KKT未达标，拒绝 |
| 未知/数组状态、NaN KKT | JSON中保存异常字段与认证问题，拒绝 |
| 求解器异常 | 保存异常类型、上下文与traceback，停止 |
| 非有限稳定LRT | 保存原始量与null表示的无效数值，停止 |
| LRT接近cutoff或原始/稳定比较相反 | 保存两者及不变cutoff，停止 |
| BIC侧歧义、校准量传播 | 即使另一比较安全也停止 |
| 正常拟合、充分统计量重算NLL | 通过，保存输入和两份拟合 |
| 映射中观测、基协方差或导数被改动 | 拒绝不成立的嵌套关系 |
| 失败运行生命周期 | startup配置、快照、failure、aborted状态和哈希完整 |
| 重复目录 | 拒绝且原文件字节不变 |
| 原计时指标 | 控制时钟证明求解与认证时间分开 |

证据位于 [unit-tests/run-001](studies/low-budget-execution-gate-001/unit-tests/run-001/)：
完整test.log、21个故障/正常fixture归档、源码快照、summary及evidence。
239个受保护历史文件运行前后哈希相同；与基线相比，冻结计划、B1、
muon_analysis和既有结果均无改动。冻结计划SHA-256：
`9185cce68f771d3245e860bc2c50de1f97509d425026cb9e12e7bb00d9ae7ffc`。
源AST核对确认六场景、null strata、重复次数、两套seed base、比较方法、
事件分配函数与生成函数均与基线一致。

## 正式运行前仍需说明

执行入口已具备本轮要求的最小闸门，可提交科研审查；本轮没有启动六场景。
单元测试不说明200次校准与六场景一定能完成，也不能估计实际歧义/失败率。
若正式run触发闸门，应审查原失败输出，不能挑选成功子集、换种子或调cutoff。
KKT和内部负曲率检查不保证全局最优；误差预算也不是严格认证界。
这些继承的数值限制仍需如实报告，不能将“通过入口测试”写成SCI性能证据。
