# 阶段0交付核验

日期2026-10-26；基线`f92eabac26092bc383bbf1e7df1731f4a93951fa`。
首版源码`493dc96db2752281f675b8f9b571de283f5c1a3a`；审查修复后canonical源码`262b743d429021493587c14f11883e52b95df2b8`，随后单独提交确定性证据与本核验记录；最终交付SHA由Git提交及交付消息给出，不写进自身以免循环。

## 1. 实际执行与结果

| 核验 | 实际结果 | 保存位置/边界 |
|---|---|---|
| 新study单元测试 | 初版12/12；审查修复后14/14通过，约0.218秒 | [stage0-tests.log](verification/stage0-tests.log)、[review-regression-green.log](verification/review-regression-green.log)；原效应预期未达仍忠实记录 |
| 既有确定性回归子集 | 11/11通过，约0.002秒 | [existing-deterministic-tests.log](verification/existing-deterministic-tests.log)；8项forward＋3项identifiability |
| canonical确定性核查 | run-002进程exit 0；状态`completed_with_failed_effect_expectation` | [run_state.json](results/run-002/run_state.json)、[checks.json](results/run-002/checks.json)；没有随机观测或新物理实验；run-001不覆盖 |
| 历史保护 | 基线1258个已有跟踪文件与基线一致，唯一允许已有文件改动是证据索引追加入口 | [archive-check.json](verification/archive-check.json)；旧源码、CSV、论文和全部历史结果未改 |
| 来源核验 | 初版9个run产物SHA-256一致、5份前序文档输入哈希一致、5个快照逐字节一致；最终两run/修复版核验另见最终归档 | [初版核验](verification/archive-check.json)；canonical [evidence.json](results/run-002/evidence.json)、[input_sources.json](results/run-002/input_sources.json) |

未运行完整旧测试套件：它包含120,000条hit-error、100,000条散射kick的大型Monte Carlo。没有把本轮25项限定检查称作“原82项全通过”。没有跑六场景、Geant4、CNN、GPU或512场景。本轮不验证优化器性能，也没有修改优化器。

## 2. 原样复现命令

在该分支仓库根目录，使用Python3与NumPy（记录环境为Python3.14.4、NumPy2.4.6、macOS arm64，见[config.json](results/run-002/config.json)）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s sci_method/studies/ml-credibility-stage0-001 -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sci_method/src:sci_method/tests \
python3 -m unittest -v test_identifiability \
  test_forward_model.ForwardModelTests.test_residual_map_matches_direct_line_extrapolation \
  test_forward_model.ForwardModelTests.test_rejects_detector_planes_out_of_declared_upstream_to_downstream_order \
  test_forward_model.ForwardModelTests.test_detector_covariance_contains_shared_upstream_correlations \
  test_forward_model.ForwardModelTests.test_path_integral_has_cross_downstream_covariance_and_zero_miss \
  test_forward_model.ForwardModelTests.test_scatter_after_first_downstream_plane_cannot_move_that_plane \
  test_forward_model.ForwardModelTests.test_scatter_between_detector_groups_keeps_causal_response \
  test_forward_model.ForwardModelTests.test_zero_material_and_zero_measurement_noise_limits \
  test_forward_model.ForwardModelTests.test_likelihood_gradient_agrees_with_central_finite_difference

# 新目录；run-001/run-002均拒绝覆盖。复现写run-003，不是自动开展新研究。
PYTHONDONTWRITEBYTECODE=1 python3 \
  sci_method/studies/ml-credibility-stage0-001/audit_checks.py \
  --output sci_method/studies/ml-credibility-stage0-001/results/run-003
```

这里列出复现命令不代表本轮执行了run-003。所有路径和残差向量均手工固定；`random_seed=null`，因为没有随机抽样，不虚构种子。config记录运行时SHA、命令和工作树状态。run-002时未跟踪results/和核验记录，不是未提交的算法改动；其源码由262b743及快照精确固定。

## 3. 数值结论与保留的失败

- 竖直和完整斜穿的两个正材料场：总covariance最大元素差均为0；Fisher rank3（按原阈值），c=v/20不可估计。解析零方向是决定性证据，非仅数值rank。
- σ=.3 mm的单投影detector covariance为[[7.74,8.82],[8.82,10.26]] mm²，完整相关项保留。
- 独立空场校准解决代数背景与目标总量的混淆，但不解决内在深度零方向；先验曲率不计入data Fisher。
- 裁切：总相对差6.84674866339067e-5，材料相对差27/676≈3.994%，总效应未达0.001的事前诊断预期。**不改门槛、不换几何、不把它写成低统计可检出的成功。**裁切rank2，不代表四层恢复。
- 64场景/128场景的风险与功效仅为闭式/有限和规划，不是实际覆盖率或ML收益。

`kernel_matches_exact=false`在裁切场景中比较的是不再适用的完整深度参考，不视为旧积分模块错误；裁切材料变化另以有理积分27/676核验。原RED与失败断言保留于[TEST_HISTORY.md](TEST_HISTORY.md)。

## 4. 科研风险与审查交接

1. 固定审查器的保证不能归功于CNN，原草案相关含混表述由新审查限定，不覆盖旧记录。
2. 一般未知nuisance下有效复合检验及ML选择后保护尚未建立，local Fisher/收敛认证不足以替代。
3. 只在已知路径、组间小角Gaussian、固定4 GeV/c模型内证明，不是Geant4或实测深度分辨率结论。
4. 两个候选假设只有潜在独立收益，均未证创新；H2尤其可能只是成熟warm start的应用。
5. 对训练集规模、风险终点和成功标准没有擅自冻结。64 calibration不能自动验证条件/OOD风险，128 test也没有通用的收益检出保证。
6. 无训练、无新实验、无合并main。最终建议B，先交ChatGPT独立科研审查，由研究者决定是否批准更小关卡。

同级Sol的范围受限只读代码审查另行记录于本study的交付审查说明；不将其冒充外部独立科研验证或本任务要求的下一轮ChatGPT审查。
