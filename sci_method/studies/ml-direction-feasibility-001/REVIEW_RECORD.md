# 案头审查记录、溯源与验收

审查日期：2026-10-23（Asia/Shanghai）。仓库：`Xuyysy/muon-imaging-reconstruction`。
固定参考提交：`809fe42be2c44e23e01e22b3c16b766803c790d8`。
独立分支：`codex/sci-ml-direction-feasibility`。

性质：静态代码/历史研究文档审查、公开文献检索、方案设计。**没有新物理数据、合成实验结果、训练权重或新算法性能结论。** 所列未来预算/种子策略待审查，不假装本轮已有随机流与复现实验命令。

## 1. 操作与隔离

原工作目录存在用户未提交的`.gitignore`/Notebook等变更，未暂存或更改。通过宿主管理worktree从固定提交建立新分支，不在原目录切换/覆盖。只添加本目录Markdown及`sci_method/RESEARCH_EVIDENCE_INDEX.md`导航入口；没有安装依赖、构建Geant4、执行研究入口、修改SCI实现或冻结指标。

技能仅用于独立工作区、后续最小实施计划和交付前核验；没有用软件技能代替科学标准、改变原实验或自动推进。未调用Astral或分派其他模型；没有修改宿主模型设置。

## 2. 阅读资产（SHA-256）

以下是实际静态核查的关键输入，不复制CSV。基于指定commit读取的其它研究文档还包括`RESEARCH_DIRECTION.md`、`RESEARCH_EVIDENCE_INDEX.md`、`README.md`和`OPTIMIZER_STABILITY_REVIEW.md`。

| 仓库根相对路径 | SHA-256 |
|---|---|
| `B1/src/PrimaryGeneratorAction.cc` | `be5df33537bf65d1ec7ad104067f3967e2d20b85454c6cc44fc4b09eb6f36961` |
| `B1/src/DetectorConstruction.cc` | `502049ad21aed21e61a86a3d04c10a07e73005ace96e577623f59a09246dc88d` |
| `B1/src/SteppingAction.cc` | `a4706ed1085bd1361cde4c7bfc2569fc52b875d519bac3755c840d710147e9dc` |
| `B1/src/EventAction.cc` | `035732afe19ccaf4f9120096863bb8ef78adb17b2e074a4421b5e6bef675dab8` |
| `sci_method/src/sci_method/forward_model.py` | `6010670daddf8cb0a90738ca40ae3d5519ad17252bbace4cf2dbcd4ffe2f7505` |
| `sci_method/src/sci_method/data_contract.py` | `423b0ff461f2e7eb29c988b3118f4015317adad3c83bd92ac3e14fd4eb994476` |
| `sci_method/src/sci_method/identifiability.py` | `7fc1bd69da3d863ae90e78650ef8423f7e45c915ad795e3549e4b7df4bb39592` |
| `sci_method/ADAPTIVE_AUDIT_FIX_RESULTS.md` | `2f463afcc021154cf63b5a50b395403131523aed7401de4b0d117e0ed5808a7b` |
| `sci_method/ADAPTIVE_AUDIT_FIX_PLAN.md` | `9185cce68f771d3245e860bc2c50de1f97509d425026cb9e12e7bb00d9ae7ffc` |
| `sci_method/LOW_BUDGET_EXECUTION_GATE_REVIEW.md` | `6b3a21f8e5702f977326a8ce2804bb980f9a3a468169be950f166bdbb58bb6ac` |
| `muon_analysis/muon_tracks.csv` | `eaa5212028a54b2c3317999461cecfafe96bb1785e22eac18e709ff8ddad0d21` |

核查命令为仓库根执行`shasum -a 256 <上述路径>`和`git ls-files 'muon_analysis/muon_tracks*.csv'`；只读，不执行分析。后者返回30个文件，不推断30个独立场景。硬件只读`sysctl -n hw.model hw.memsize hw.physicalcpu`返回Mac17,4 / 17179869184 / 10；该命令不是吞吐测试。

## 3. 检索问题、路径与取舍

检索包含如下原始关键词组合（在同一审查日）：

- `muon tomography MINT neural implicit field reconstruction`
- `2606.20180 muon`，随后分别打开arXiv v1/v2方法和讨论。
- `muon tomography physics informed deep learning reconstruction uncertainty 2024 2025`
- `muon scattering tomography U-Net 2026 denoising short time`
- `muon tomography transformer deep unfolding reconstruction 2024 2025 2026`
- `muon unrolled reconstruction`、`muon tomography deep unfolding`、`muon learned primal reconstruction`
- `muon tomography Bayesian Gaussian scale mixture Wang Schultz Qi 2009`
- `On hallucinations in tomographic image reconstruction null space`
- `learned null space networks data consistency uncertainty inverse problems 2023 2024`
- `conformal inverse problems`及TaskUQ、DPS原始论文。
- `muon adaptive mesh reconstruction 2025`、`muon tomography adaptive mesh refinement`
- `gamma spectroscopy domain adaptation PyRIID low count 2024 2025 detector`
- `Geant4 G4ParticleGun SetParticleEnergy kinetic energy`、Geant4官方μ⁻定义。

结果逐篇指向[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)的M1–M13、G1–G8，那里保留原始论文URL、版本、状态及阅读层级。另以`low count PET learned prior unrolled reconstruction Poisson likelihood 2024 2025 arxiv`补查低计数学习先验。搜索中出现的非原研究聚合站、无关ATLAS跟踪/吸收层析、优化器同名Muon网页不作为方法事实依据。

关键正文审查位置：

| 来源 | 实际获取/审查重点 | 限制 |
|---|---|---|
| MINT出版社索引/官方GitHub | §2.1–2.5似然/神经场、§4实测、§5比较范围/正则与信息讨论；[官方仓库README](https://github.com/dr4kan/MINT) | 直接页面部分403；通过出版社检索索引读相关正文，不当作完整PDF/代码审计。后续检索已找到官方repo并核实公开MIT标识、依赖和输入单位；LICENSE/pyproject原文件抓取失败，未运行代码 |
| RHMT v1 / NAMT v2 | v1 §3分支、v2 §3公式(1)–(7)、§4/6投影评价和局限 | 同arXiv记录，只计一项；未复现实验/代码 |
| U-Net arXiv v1 | §2.1–2.3数据/损失、§3泛化；abs journal reference | 原文“11 independent”描述不被本报告沿用为11次独立采集；600k母池/相同C形数据需分层解释 |
| JINST structural diagnostics arXiv v2 | §3/4几何、统计矩、增强、原几何级test划分 | 与出版社终稿逐字差异未核对；不抹去其已经实施的几何留出 |
| Wang Bayesian/GSM | 公开学术机构PDF的方法与评价段 | 仅链接、不复制全文；期刊受版权保护不因private仓库就有上传权 |
| G1/G2/G3 | 广义零空间定义、数据邻近/不确定损失与应用 | 线性CT/MRI原理不能无证明照搬到随机协方差逆问题 |
| TaskUQ / DPS | 区间/边际与条件覆盖、likelihood近似和限制 | 本轮不声称获得缪子OOD覆盖或精确posterior sampler |
| 低计数PET G8 | §II Poisson模型、§IV likelihood schedule、§V MAP/展开式对照和数据划分 | 它说明数据一致性/似然水平与学习先验已有结合；不是缪子方法或本项目训练资源 |
| Gamma domain adaptation v3 | §3数据、§4适配/ensemble与§5比较 | 有实测的文献不是本项目有权使用的数据；不同分类/定量任务不混排名 |
| Transformer/Mamba预印本线索 | 国家预印本摘要索引、ChinaXiv原链接及核物理分站索引 | 不能访问原全文；不把chinarxiv.org镜像当作者原始方法证据；M13标P/U，须追查 |

本轮没有复制第三方论文PDF、图表、完整文章或代码进入仓库；未写Zotero库或下载受限文献。没有穷尽审查所有2026新文献，未取得全文之处不能用于“别人没有做过”的结论。

## 4. 发现与未决风险

- MINT是最接近B1的直接竞争者；NAMT已覆盖B3的大块叙述；B2的通用核心已有null-space/UQ强近邻。推荐只是条件性探索，不是查新认证。
- 当前主分支资产不能证明同primary四层hit身份一致；旧文档PDG符号错误、T/p方差方向歧义在新文档指出，原记录保留。
- 旧原型有数值失败与合成开发证据边界；不将软件测试提升为SCI性能结果。
- 真实数据、模型迁移、profile全局优化、近邻代码可复现性、实际算力吞吐均未确定。没有用预期收益填补这些缺口。

## 5. 交付核验

已执行的交付核验：6份本目录Markdown、12个本地相对链接全部存在；11项关键输入SHA-256全部匹配；差异集合严格等于这6份文件＋1个证据索引；保护范围与基线无差异，`git diff --cached --check`通过。算法/数据未改，因此本轮不重新运行82项数值单元测试；其历史通过记录仅引用旧报告，不声称本轮重跑。检查只读解析Markdown、比对Git差异与hash，没有运行研究模块。

输出文件哈希可用`shasum -a 256 sci_method/studies/ml-direction-feasibility-001/*.md`读取，Git commit提供不可变内容标识；无需将包含自身的hash写入自身。最终提交/推送结果以终端和交付消息为准。
