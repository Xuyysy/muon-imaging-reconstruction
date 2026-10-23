# 竞争文献审查与创新性边界

检索截止：2026-10-23。定向审查，不是穷尽性系统综述或“查无先例”证明。以下均为原研究/作者发布渠道；检索摘要用于发现线索，不以搜索引擎排名判断质量。

阅读层级：**M**=已获取方法/实验/限制的正文相关章节，非声称逐字通读全部附录；**P**=出版社页面/摘要/元数据；**C**=会议摘要；**U**=尚未核实。M 也不等于复现。出版社部分页面直接访问 403，MINT 的方法章节通过其出版社页面的检索索引返回，需在正式复现前核对固定 PDF 版本。

## 1. 缪子重建竞争矩阵

各行的数据量是文献自身设置，不是本项目所需训练量。未知处明确保留未知。

| ID / 论文与状态 | 数据来源及需求 | 创新/物理约束 | 监督方式 | 泛化与可信度审查 | 对我们的影响 |
|---|---|---|---|---|---|
| M1 Pagano, [MINT](https://doi.org/10.1016/j.nima.2026.171923)，NIM A online 2026 / 卷1093，2027；M，§2、4、5 | 两个 Geant4 基准＋实测；逐扫描优化，不需跨场景标签训练集 | hash-grid 神经场；逐径迹 Highland Gaussian NLL；TV/背景约束 | 无监督物理似然拟合，不是跨场景监督网络 | 正文有网格/预算压力测试、正则消融；实测子样本离散不等于覆盖率。论文限定比较为其未正则 PoCA/MLEM，而非所有 MAP | B1 直接近邻。不能把 neural field、物理 NLL 或多尺度本身当新贡献；需同等正则强基线。[官方仓库](https://github.com/dr4kan/MINT)已核实README/公开MIT标识，未运行/逐式审计 |
| M2 Zhao 等，[RHMT v1](https://arxiv.org/html/2606.20180v1) → [NAMT v2](https://arxiv.org/html/2606.20180v2)，2026，同一预印本，M，§3–6/附录 | Geant4＋CRY；v2 八种 U 形材料/背景组合；空场校准 | v1 投影轨迹 nuisance、FE 协方差及磁谱仪能损分支；v2 三/四层 residual、共享动量尺度积分、条件空场响应＋TV | 物理似然＋空场响应拟合；非目标图像监督 | v2 保持 U 形、改变材料/误差/预算；报告深度投影；式(6)给定共享尺度后坐标因子化，不等于我们完整交叉协方差。未据此判断谁更优 | B3 直接近邻；相关误差/结构置信度需实质比较。两版只计一项工作；无已核实期刊接受信息 |
| M3 Wang 等，[U-Net enhancement](https://arxiv.org/html/2602.07060v1)，2026；NIM A1092:171797，[DOI](https://doi.org/10.1016/j.nima.2026.171797)；M，§2–3 | Geant4/CRY C 形目标、600k 事件母池；实验八层 micromegas | PoCA 图像增强＋Stamping 域噪声混合；非原始事件似然约束 | 高统计 PoCA 为标签；不是材料真值 | 有泛化/消融。实验18,417事件的11个10k子集不能是11次独立物理采集；仿真不同子样本/噪声不等于独立形状。不能概括为“完全没有泛化” | “PoCA→U-Net”、sim-to-real 风格注入均非新点；需追踪母场景/事件重叠，不照搬指标主张 |
| M4 Pezzotti 等，[structural diagnostics](https://arxiv.org/html/2502.03339v2)，JINST20 P06034，2025；M，§3–5 | Geant4 铁筋/混凝土；100原始几何、旋转/裁剪扩充；15种统计量图 | U-Net＋RRDB，输入径迹统计矩图 | 几何标签/高统计材料表征 | 正文明确原几何1–50测试、51–100训练；不能声称事件随机划分或无几何隔离。相关增强不创造新独立场景 | 仅加入几何留出也不是我们的独占创新；要做覆盖/先验冲突/结构置信度 |
| M5 Vinodkumar 等，[3D CNN denoising](https://link.springer.com/article/10.1007/s11760-025-03992-1)，SIVP19:378，2025；P | 出版页称6目标、两材料三尺寸；完整数据/切分未核实 | PoCA三维体素＋3D CNN | 监督细节未核实 | 订阅正文未获取；数据声明与摘要存在需进一步核对之处，不据此判论文有效/无效 | 已足以否定“首次PoCA＋3D CNN”；不能据摘要排除其可信度/OOD工作 |
| M6 [ML-based dual-view localization](https://doi.org/10.1109/ACCESS.2025.3648859)，online2025、Access14，2026；P | 直行/散射径迹，精确场景量未知 | 双视图二值定位网络 | 正负目标印迹监督 | 未取得完整方法，未核实校准与分布外范围 | 定位任务近邻；不能与物理散射密度直接比分类准确率 |
| M7 Alameddine 等，[gradient-descent reconstruction](https://arxiv.org/html/2511.05226v1)，MODE2025 contribution/预印本；M，§2–3 | 模拟径迹；无需监督图像集 | voxel散射 angle–displacement Gaussian NLL 自动微分 | 逐对象极大似然；使用PyTorch不等于学习先验 | 展示模拟重建，未确认独立可信度校准 | 单纯“可微重建/用PyTorch”不构成 ML 创新 |
| M8 Wang/Schultz/Qi，[Bayesian MAP](https://doi.org/10.1109/TIP.2009.2014423)，TIP18:1080–1089，2009；M，[正文](https://www.hep.phy.cam.ac.uk/~hommels/CosmicConcrete_dir/MuonPapersForBart/Paper_IEEE_18_5_May2009_BayesianImageReco.pdf) | 模拟与ROC检测研究 | 先验收缩/正则MAP；散射密度进入协方差 | 手工先验，无神经训练 | 检测评价不等于频率覆盖校准 | 必须比较强 MAP；不能把正则化减少方差当成新增数据可辨识性 |
| M9 Wang/Schultz/Qi，[GSM](https://doi.org/10.1109/TNS.2009.2023518)，TNS56:2480–2486，2009；M，[正文](https://www.hep.phy.cam.ac.uk/~hommels/CosmicConcrete_dir/MuonPapersForBart/Paper_IEEE_56_4_Aug2009_GaussianMixModel_for_BayesianImageReco.pdf) | 模拟/LROC | Gaussian scale mixture重尾＋MAP/MM | 统计估计 | 已研究高斯尾失配；不是现代网络，但仍是强对手 | B3 不可把“混合模型处理非高斯尾”当新颖点；与M8是两篇不同工作 |
| M10 Schultz 等，[statistical reconstruction](https://doi.org/10.1109/TIP.2007.901239)，TIP16，2007；本轮P/旧索引 | 经典统计重建 | MLS-EM/多重散射统计模型 | 物理估计 | 本轮未重新获取完整正文，不补写实现细节 | 必选基线；若已知动量、路径、正则不同需注明，不假称同一目标函数 |
| M11 [大体素/角截断](https://doi.org/10.1016/j.nima.2024.169932)，NIM A2024；本轮P/旧索引 | 仿真/实验条件需取得正文核实 | 降低体素波动的PoCA改进 | 非学习 | 不用摘要断言其没研究自适应 | 粗网格/角截断不是算法创新；不能当唯一强基线 |
| M12 Wu，[DQN adaptive reconstruction](https://indico.wigner.hu/event/1727/attachments/3207/6643/20260430_Muographers2026_AbstractBook.pdf)，Muographers2026摘要，C | 摘要描述PoCA＋3D CNN状态与受限动作；规模未知 | 物理边界内参数选择/RL | 奖励依赖图像质量，详细定义未知 | 没有已核实完整同行评审方法/复现证据 | B4已有直接线索；既不能称已解决，也不能以未取全文认定空白 |
| M13 [MuTrans-C/MuMamba-C原记录](https://chinaxiv.org/abs/202609.00116)，2026-09-09预印本；P/U，[国家平台摘要索引](https://sinoxiv.napstic.cn/article/26205404) | 索引摘要称稀疏径迹到三维场；完整数据需求未知 | 摘要称Transformer/Mamba encoder、cross-attention decoder，细节未核实 | 未核实 | 原记录直接访问失败，仅核实索引元数据/摘要；不以第三方镜像作为已读原文，不列已验证强基线 | Transformer方向已有需追查线索；不能由此断言具体方法已解决问题，也不能宣称空白 |

## 2. 通用可信逆问题与改题参照

| ID / 来源与状态 | 数据/方法/监督 | 已经研究什么 | 我们仍需解决的区别与边界 |
|---|---|---|---|
| G1 Bhadra等，[hallucinations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8673588/)，TMI40:3249–3260，2021；M，定义与讨论 | 线性成像测量/零空间分解、幻觉图 | 先验可产生观测不支持的结构，区分广义测量/零空间 | 原理不是新发现。缪子是随机协方差观测，不能把确定性CT的残差投影公式直接套用 |
| G2 Angermann等，[uncertainty-aware null-space networks](https://arxiv.org/html/2304.06955v1)，2023预印本；M，§2.3–3 | MRI/有限角CT；零空间网络＋输入依赖Laplace尺度图；监督训练 | 数据一致性与学习不确定度已结合，有真实MRI数据 | B2不能只加Fisher图/方差头。需散射联合协方差/nuisance、结构级校准、等分布反例；学习尺度不自动成为频率置信区间 |
| G3 Göppel等，[data-proximal null-space networks](https://arxiv.org/html/2309.06573v1)，2023预印本；M，§2 | 线性算子、数据邻近约束、收敛分析、有限角CT | 有界观测改动＋学习零空间补全已有理论 | 完全一致的零空间补全仍可错误。我们的任务是明确拒判，而不把一致性当成结构真实性证明 |
| G4 Wen等，[TaskUQ](https://arxiv.org/html/2405.18527v2)，2024预印本版本；M，§2–3 | MRI posterior samples＋独立calibration；任务输出区间 | 结构/任务不确定度、conformal边际覆盖、多轮测量 | 不能宣称首次任务级UQ；exchangeability不含任意OOD条件覆盖。对缪子须按场景校准，而非以轨迹作独立校准对象 |
| G5 Chung等，[DPS](https://arxiv.org/html/2209.14687v3)，ICLR2023；M，§3/限制 | 学习图像score先验＋近似posterior sampling；含Gaussian/Poisson噪声 | 可微物理模型＋强学习先验已有成熟框架 | 不代表准确后验/置信覆盖。缪子给定N的残差似然不是低计数Poisson图像模型；扩散先验数据与算力成本高，不作首个原型 |
| G6 Lalor等，[gamma sim-to-real](https://arxiv.org/html/2412.07069v3)，NIM A，[DOI](https://doi.org/10.1016/j.nima.2025.171159)，2025；M，§3–5 | GADRAS/Geant4及LaBr3、NaI实测；MLP/CNN/Transformer和小样本微调 | 已覆盖多探测器/域适配，区分混合比例与单标签任务 | γ改题并非低竞争捷径。需响应/核素模板/真实谱授权；当前缪子kernel和训练样本不能直接复用 |
| G7 [Sandia PyRIID](https://github.com/sandialabs/pyriid)，官方开源项目，非单独性能论文；README已读 | 谱合成、识别、估计工具及文献链接 | 有现成核素ML生态与强基线 | 可降低软件入门成本，不提供本项目自动拥有的真实跨探测器验证数据 |
| G8 Webber等，[Likelihood-scheduled score-based PET](https://arxiv.org/html/2412.04339v1)，2024预印本版本；M，§II/IV/V | 低计数Poisson PET；预训练score先验＋MLEM似然调度；模拟及真实3D数据 | 将学习先验与physics update结合、降低调参；比较MAP-EM及展开式FBSEM-net；55训练/1验证/13测试的3D数据划分 | 同等likelihood值仍对应多个图像，不意味着结构都真实。属于均值Poisson逆问题，不可把PET损失/计数直接搬到缪子协方差观测；先验调度/深度展开不是独占创新 |

## 3. 路线与“已被研究”对应

- **B1**：M1/M7/G5/G8直接覆盖物理可微、神经表示和likelihood-guided学习；“加物理loss”不足。G8明确给出低计数逆问题中的学习先验和展开式强对照，但不是缪子专用方法。
- **B2**：G1–G4覆盖大部分通用可信思想。潜在空间是协方差逆问题的局部结构可估计性与nuisance混淆下，学习补全的校准/拒判；其新颖性置信度只为**中低**，不是查新通过。
- **B3**：M2/M9已处理空场、动量和重尾；学习全协方差/条件失配需胜过GSM、Student-t和NAMT，不只是比单高斯好。
- **B4**：M1多分辨率、M12强化学习调参已有线索。收益可能是算力节约；不能把计算分配当额外物理信息，也不能控制自然缪子按希望路径入射。
- **γ改题**：G6/G7表明可复用生态强，但新颖性、授权和真实响应数据门槛同样高。

## 4. 明确未完成的查新

未核实到可直接运行、带完整协议的缪子**深度展开**强实现；定向检索“muon tomography + deep unfolding / unrolled / learned primal”未形成可靠条目。这只表示本次检索缺口，不代表不存在论文。ATLAS/MUonE的Transformer轨迹识别、吸收型muography与散射密度重建任务不同，不混入性能排名。

正式算法立项前还需：M1固定论文/代码commit并审查实现、输入适配及LICENSE原文；M13全文；M5/M6方法和数据划分；追踪G2–G4的后续文献、B2与已有不确定度方法的逐式差异；基于参考文献前后向检索补充自适应网格/深度展开。只要这些近邻中已有完整同等方案，应降级为复现/审计研究，不通过改名称继续宣称新算法。

MINT可获取性补查：官方README要求Python≥3.10、PyTorch≥2.0，声明纯PyTorch支持CPU/CUDA/MPS。输入位置以cm、动量以MeV/c，输出密度cm^-1；与本项目mm、GeV/c、mm^-1不同。其entry/exit位置和边界约定需审查适配，不能无单位转换直接读旧CSV。仓库存在不等于本机可复现，更不意味着实测数据随代码开放；本轮未安装或执行。

纠正文献标识：旧对话中的 `10.1016/j.nima.2026.005231` 不是本次核实的 U-Net DOI；`S0168900226005231` 是 PII，对应论文DOI为 `10.1016/j.nima.2026.171797`。不覆盖历史对话/资料。
