# SCI 新算法研究证据索引

审查基线：`af310a86f65d3be43b7945d7176d17ff981314c5`。此索引指向仓库现存资料，不复制、不重写原始科研数据；所有已查看旧场景对新算法均不是未见测试集。

## 1. 共享系统、观测数据与源代码

| 证据 | 路径 | 研究用途 | 当前阶段/限制 |
|---|---|---|---|
| Geant4 输入生成与 source mode | [`B1/src/PrimaryGeneratorAction.cc`](../B1/src/PrimaryGeneratorAction.cc) | 理想 4 GeV 平行源；另有半自然近似开关 | 源模式应以代码提交为准；半自然模式不是经验证的 cosmic spectrum |
| detector geometry | [`B1/src/DetectorConstruction.cc`](../B1/src/DetectorConstruction.cc) | 四层位置探测器与目标几何 | 位置误差、对准、效率、读出离散化不构成完整真实探测器模型 |
| event writer | [`B1/src/EventAction.cc`](../B1/src/EventAction.cc) | 四层击中和派生 PoCA/散射角字段 | 仅完整四层事件写出；共享输出有副作用 |
| 原始四层候选数据 | [`muon_analysis/muon_tracks.csv`](../muon_analysis/muon_tracks.csv) | 新算法观测输入；仅四层 hit 字段可用 | 9,975 行；SHA-256 `eaa5212028a54b2c3317999461cecfafe96bb1785e22eac18e709ff8ddad0d21`；Geant4 run/seed/exact command 未知；只作 development |
| 来源清单 | [`muon_analysis/RESOURCE_MANIFEST.json`](../muon_analysis/RESOURCE_MANIFEST.json) | 资产哈希、行数、已知/未知 provenance | 已记录主要基线 CSV 的 RNG 与准确复现命令未知 |

## 2. 旧算法与结果

| 资源 | 路径 | 可用于 | 不可用于 |
|---|---|---|---|
| 旧 PoCA baseline 实现 | [`muon_analysis/run_baseline_experiment.py`](../muon_analysis/run_baseline_experiment.py) | 方法说明及历史参照；PoCA 固定网格输出复查 | 新算法的真值隔离推断；其评估函数包含真值 mask 与 oracle Top-k，runner 固定写路径 |
| 固定体素基线结果 | [`muon_analysis/baseline_results/`](../muon_analysis/baseline_results/) | 已查看的开发基准 | 新方法冻结验证或最终未见测试 |
| 旧自适应体素实验 | [`muon_analysis/run_adaptive_voxel_experiment.py`](../muon_analysis/run_adaptive_voxel_experiment.py)、[`muon_analysis/adaptive_voxel_results/`](../muon_analysis/adaptive_voxel_results/) | 说明旧“按 PoCA score 选部分格子细分”基线及历史正/负结果 | 宣称统计似然新贡献；独立验证 |
| 旧误差注入 | [`muon_analysis/run_detector_resolution_robustness.py`](../muon_analysis/run_detector_resolution_robustness.py)、[`muon_analysis/detector_resolution_results/`](../muon_analysis/detector_resolution_results/) | 审查高斯 hit x/y 扰动流程及历史趋势 | 真实探测器模型/完整误差预算；独立 run；新算法最终测试 |
| 事件不确定度/角度处理 | [`muon_analysis/event_uncertainty_results/`](../muon_analysis/event_uncertainty_results/)、[`muon_analysis/uncertainty_aware_angle_results/`](../muon_analysis/uncertainty_aware_angle_results/) | 历史误差传播与修正候选 | 已证明的统计似然或新算法贡献 |
| 跨场景旧研究 | `muon_analysis/*_results/` 与对应 `*_PLAN.md`/`research_note.md` | 作为开发背景、阴性结果和复用警示 | 任何新算法独立测试；目录重命名不会建立新数据 |
| 旧论文工作稿 | [`B1/paper/main.tex`](../B1/paper/main.tex)、[`B1/paper/README.md`](../B1/paper/README.md) | 追溯既有科学叙述和限制 | 本 SCI 新算法论文；不能直接把旧结论改写成新方法结果 |

## 3. 数据隔离登记

| 数据组 | 已知来源情况 | SCI 新方法角色 |
|---|---|---|
| `muon_tracks.csv` | 被旧 POCA/自适应/测量误差相关代码和论文查看；Geant4 seed/run 未确证 | 开发、输入契约/数值管线测试；绝非独立测试 |
| 其他 `muon_tracks*.csv` | 已纳入资源清单，多数为既有实验/对照；必须按清单和对应 `*_results` 核验 exact macro、seed 与事件重叠 | 默认开发/历史对照；不得按新算法重新命名为 holdout |
| 五次子抽样 | 旧实验从同一事件 CSV 派生；不等于五个 Geant4 独立物理 run | 可作同一输入下的抽样稳定性分析；统计单位需明确 |
| 新未来生成数据 | 尚不存在于本轮 | 先冻结 protocol 后才可生成；分离 development、tuning、frozen validation、final unseen |

## 4. 文献检索记录（2026-10-19）

本轮只做定向检索，非系统综述；外部全文未复制到仓库。引文均链接至 DOI、出版社或 arXiv 原始页面。

| 文献 | 相关性 | 当前阅读结论 |
|---|---|---|
| Schultz et al. (2007), [Statistical reconstruction for cosmic ray muon tomography](https://doi.org/10.1109/TIP.2007.901239) | MLS-EM 强基线 | 已提出多重散射统计似然/EM与实验误差稳健性；模型似然化不是新颖点 |
| [Multiparametric approach…](https://doi.org/10.1140/epjp/s13360-020-00970-z) (2020) | POCA 基线限制与真实 detector | 明确 PoCA 是单点简化；报告边界定位、屏蔽和小统计波动问题 |
| [A new efficient imaging reconstruction method…](https://doi.org/10.1016/j.nima.2024.169932) (NIM A, 2024) | 低统计量/大体素 | 大体素结合散射角截断降低 voxel fluctuations；角度截断+粗网格不可作为独立创新 |
| [TomOpt detector design and volume inference](https://doi.org/10.1063/5.0287758) (JAP, 2025) | PoCA uncertainty | 以 PoCA 三维不确定度分布跨 voxel 加权，并做体素辐射长度/聚类评分；概率扩散 PoCA 不是新空白 |
| Ughade & Chatzidakis, [μTRec](https://doi.org/10.1063/5.0278370) (JAP, 2025) | Bayesian trajectory model | 报告贝叶斯高斯径迹/MCS/能损轨迹和低统计量场景；详细代码及假设待核实 |
| Alameddine et al., [Gradient-descent-based reconstruction…](https://arxiv.org/abs/2511.05226) (2025 preprint) | 可微 likelihood | 直接用 PyTorch 自动微分优化 voxelized scattering likelihood；梯度优化框架不是新颖点 |
| Zhao et al., [Raw-Hit Muon Tomography / Nuisance-Aware Muon Tomography](https://arxiv.org/abs/2606.20180) (arXiv 2026) | 最直接近邻 | 使用四平面下游 hit residual、路径积分散射核、open-field nuisance 响应、动量边缘化；覆盖 hit-error 和低-count。标题/版本须核对，但内容对本方向高度重合 |
| Pagano, [A differentiable neural-field forward model for cosmic-ray muon scattering tomography](https://doi.org/10.1016/j.nima.2026.171923) (online 2026 / vol. 2027) | 最新连续场近邻 | 多分辨率 hash-grid 表示 + 每轨迹 Highland NLL + TV/background 正则，含低统计量和实测数据；神经场/多尺度似然不能单独作为贡献 |

## 5. 图像/结论来源规则

- 本分支没有新实验图、CSV、论文性能数字或 `evidence.json`；没有运行新 Geant4/算法实验。
- 新算法后续的每个 study 应位于 `sci_method/studies/<study-id>/`，各自保存协议、配置、输入哈希、代码版本、物理与分析随机流、复现命令、日志、结果和失败记录。
- 论文图必须链接其生成脚本、study ID、源数值表与 `evidence.json`；没有 evidence 元数据的图不得用作正式定量主结论。
- 原始输入仍保持 `muon_analysis/` 唯一存放，不复制到 SCI 子目录；对共享 CSV 只读。
