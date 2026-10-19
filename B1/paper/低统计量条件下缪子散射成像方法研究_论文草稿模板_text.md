# Document text extraction: 低统计量条件下缪子散射成像方法研究_论文草稿模板.docx

Source: `B1/paper/低统计量条件下缪子散射成像方法研究_论文草稿模板.docx`

Source SHA-256: `b8fb8dd670ba778259a2a6f55476c9f23f3a77c74d3b891bb5bc8febcd607023`

Paragraphs: 98; top-level tables: 4

Extraction: `python-docx 1.2.0`. Paragraphs and tables follow document-body order. Page numbers are omitted because Word pagination varies by renderer; the DOCX remains authoritative for layout and graphics.

### Paragraph 2 · Title

```text
低统计量条件下缪子散射成像方法研究
```

### Paragraph 3 · Paper Subtitle

```text
论文草稿模板
```

### Table 1

#### Row 1

Cell 1
```text
作者
```

Cell 2
```text
【填写姓名】
```

#### Row 2

Cell 1
```text
导师
```

Cell 2
```text
【填写导师姓名】
```

#### Row 3

Cell 1
```text
专业或研究方向
```

Cell 2
```text
缪子成像与重建算法
```

#### Row 4

Cell 1
```text
版本
```

Cell 2
```text
v0.1  基线研究草稿
```

#### Row 5

Cell 1
```text
日期
```

Cell 2
```text
【填写日期】
```

### Paragraph 6 · Heading 1

```text
写作说明
```

### Paragraph 7 · Body Text

```text
本草稿用于伴随仿真、重建和算法实验持续更新。当前已写入的内容对应项目的已知基线：四层探测器、U 型目标、Geant4 事件输出、Python POCA 成像，以及理想和半自然缪子入射两种模式。所有带“待补充”的位置应在实验推进时逐步替换为具体参数、图像和结果。
```

### Paragraph 8 · Body Text

```text
建议每完成一次可复现实验，就同步补充实验配置、随机种子或版本号、输出文件位置和一张关键图。这样论文中的结论始终能追溯到对应的数据与代码版本。
```

### Paragraph 9 · Heading 1

```text
摘要
```

### Paragraph 10 · Body Text

```text
【待补充】概述研究背景、研究问题、提出的方法、核心实验条件、最主要的定量结果和结论。摘要应在结果稳定后完成，建议控制在 300 至 500 字。
```

### Paragraph 11 · Heading 1

```text
关键词
```

### Paragraph 12 · Body Text

```text
【待补充】例如：宇宙线缪子；缪子散射成像；POCA；低统计量；自适应体素；事件筛选。
```

### Paragraph 13 · Heading 1

```text
1 引言
```

### Paragraph 14 · Heading 2

```text
1.1 研究背景与意义
```

### Paragraph 15 · Body Text

```text
缪子散射成像利用带电缪子穿过物体时产生的多重库仑散射，实现对被测区域内部高散射材料或结构的非侵入式探测。相较于主动辐射成像方法，它能够利用天然宇宙线缪子，在大尺寸、厚屏蔽或不便接触的场景中具有应用潜力。
```

### Paragraph 16 · Body Text

```text
【待补充】结合你的具体应用场景，补充核设施、工程结构、货物检测或其他研究对象，并插入文献引用。
```

### Paragraph 17 · Heading 2

```text
1.2 研究问题
```

### Paragraph 18 · Body Text

```text
在有限探测时间或低统计量条件下，传统 POCA 重建容易出现散射点分散、背景噪声较高和目标边界模糊等问题。本研究关注如何在保持物理可解释性的前提下，提高低统计量条件下的目标定位与轮廓重建质量。
```

### Paragraph 19 · Heading 2

```text
1.3 研究目标与贡献
```

### Paragraph 20 · List Bullet

```text
建立四层探测器与 U 型目标的 Geant4 缪子散射成像仿真基线。
```

### Paragraph 21 · List Bullet

```text
实现基于入射和出射轨迹的 POCA 三维重建，并建立可重复的定量评价流程。
```

### Paragraph 22 · List Bullet

```text
在不同事件数量下研究传统 POCA 的成像退化规律。
```

### Paragraph 23 · List Bullet

```text
提出并验证角度筛选和自适应体素等低统计量重建改进方法。
```

### Paragraph 24 · List Bullet

```text
在上述物理基线基础上，探索机器学习事件筛选或图像去噪的增益与泛化能力。
```

### Paragraph 25 · Heading 2

```text
1.4 论文结构
```

### Paragraph 26 · Body Text

```text
第 2 章综述相关方法；第 3 章介绍仿真系统与数据；第 4 章给出 POCA 基线；第 5 章说明实验设计和评价指标；第 6 章提出改进方法；第 7 章给出结果与讨论；第 8 章总结全文。
```

### Paragraph 27 · Heading 1

```text
2 相关工作
```

### Paragraph 28 · Heading 2

```text
2.1 缪子散射成像方法
```

### Paragraph 29 · Body Text

```text
【待补充】梳理 POCA、角度统计重建、MLEM、最大似然或其他代表性方法。每一类方法至少引用 1 至 3 篇核心文献，并说明其优势与局限。
```

### Paragraph 30 · Heading 2

```text
2.2 低统计量成像与体素策略
```

### Paragraph 31 · Body Text

```text
【待补充】总结大体素、分层或自适应体素、散射角截断、先验约束等工作；明确本文与已有工作的差异。
```

### Paragraph 32 · Heading 2

```text
2.3 机器学习在缪子成像中的应用
```

### Paragraph 33 · Body Text

```text
【待补充】关注事件筛选、图像去噪、材料分类和端到端重建。说明本研究将机器学习作为物理方法的辅助模块，而不是替代物理基线。
```

### Paragraph 34 · Heading 1

```text
3 缪子散射成像仿真系统与数据
```

### Paragraph 35 · Heading 2

```text
3.1 系统总体流程
```

### Paragraph 36 · Body Text

```text
本研究的处理流程为：Geant4 生成缪子事件；探测器记录上下游击中信息；Python 读取 CSV 数据并拟合入射与出射轨迹；根据轨迹计算散射角和 POCA 位置；最后将散射事件累积到三维体素中形成图像。
```

### Paragraph 37 · Normal

```text
图像或示意图预留位置
```

### Paragraph 38 · Normal

```text
图 X  缪子成像仿真与重建流程图
```

### Paragraph 39 · Heading 2

```text
3.2 探测器与目标几何
```

### Paragraph 40 · Body Text

```text
当前基线采用四层平面探测器，其中两层位于目标上游、两层位于目标下游。被测对象设置为 U 型材料结构，以便直观观察目标轮廓、空腔和边缘在散射图像中的表现。
```

### Paragraph 41 · Body Text

```text
【待补充】填写探测器平面尺寸、层间距、目标材料、U 型几何尺寸、世界体尺寸及坐标系定义。建议配一张 Geant4 几何示意图。
```

### Paragraph 42 · Heading 2

```text
3.3 缪子源模型
```

### Paragraph 43 · Body Text

```text
为区分算法基线与环境因素，当前代码中设置了两种可切换的缪子入射模式。理想模式使用固定 4 GeV、沿正 z 方向的平行缪子束，便于验证几何与 POCA 流程。半自然模式使用随机入射位置、均匀方位角、近似 cos²(theta) 天顶角分布以及 2 至 8 GeV 的简化能量范围，用于初步评估入射角和能量变化对重建的影响。
```

### Paragraph 44 · Body Text

```text
半自然模式是教学和敏感性分析用的中间模型，不应被表述为精确的天然宇宙线缪子谱。后续若开展贴近实验条件的研究，应考虑引入 CRY、EcoMug 或经过验证的海平面缪子通量模型。
```

### Paragraph 45 · Heading 2

```text
3.4 事件数据与 CSV 字段
```

### Paragraph 46 · Body Text

```text
每个有效事件应至少保存四层探测器的击中坐标、事件编号以及必要的筛选信息。Python 分析程序基于这些字段构造上游和下游直线，从而计算散射角与 POCA 点。
```

### Paragraph 47 · Body Text

```text
【待补充】插入 CSV 字段表，并说明单位、缺失值处理方式、完整径迹的判定条件和数据保存目录。
```

### Paragraph 48 · Heading 1

```text
4 POCA 基线重建方法
```

### Paragraph 49 · Heading 2

```text
4.1 上下游轨迹拟合
```

### Paragraph 50 · Body Text

```text
对于每个具有完整四层击中的事件，利用上游两层击中点拟合入射直线，利用下游两层击中点拟合出射直线。两条直线的方向变化反映了缪子穿过目标后的散射程度。
```

### Paragraph 51 · Heading 2

```text
4.2 散射角与 POCA 点
```

### Paragraph 52 · Body Text

```text
散射角由入射方向和出射方向之间的夹角计算。POCA 方法将两条空间直线的最近点对的中点近似视为主要散射位置，再根据散射角或角度权重累积到三维体素。
```

### Paragraph 53 · Body Text

```text
【待补充】在此插入 POCA 的数学定义、直线参数方程和体素累积公式。若使用角度权重，应说明具体权重函数。
```

### Paragraph 54 · Heading 2

```text
4.3 基线实现与可视化
```

### Paragraph 55 · Body Text

```text
基线输出应至少包含 xy、xz、yz 三个投影图或三维点云图，并显示目标区域、体素尺寸、散射角筛选范围和事件数量。所有图应保存对应的脚本版本和输入 CSV 文件名。
```

### Paragraph 56 · Normal

```text
图像或示意图预留位置
```

### Paragraph 57 · Normal

```text
图 X  U 型目标的 POCA 基线重建结果
```

### Paragraph 58 · Heading 1

```text
5 实验设计与评价指标
```

### Paragraph 59 · Heading 2

```text
5.1 对照组与实验变量
```

### Paragraph 60 · Body Text

```text
实验以传统 POCA 为对照组，逐步比较角度筛选 POCA、自适应体素 POCA，以及后续可选的机器学习事件筛选或去噪方法。每次比较只改变一个核心变量，其他几何、材料和随机设置保持一致。
```

### Table 2

#### Row 1

Cell 1
```text
实验编号
```

Cell 2
```text
方法
```

Cell 3
```text
改变因素
```

Cell 4
```text
主要输出
```

#### Row 2

Cell 1
```text
E0
```

Cell 2
```text
传统 POCA
```

Cell 3
```text
基线参数
```

Cell 4
```text
重建图像与定量指标
```

#### Row 3

Cell 1
```text
E1
```

Cell 2
```text
角度筛选 POCA
```

Cell 3
```text
散射角阈值
```

Cell 4
```text
信噪比与轮廓变化
```

#### Row 4

Cell 1
```text
E2
```

Cell 2
```text
自适应体素 POCA
```

Cell 3
```text
粗细体素策略
```

Cell 4
```text
定位精度与计算时间
```

#### Row 5

Cell 1
```text
E3
```

Cell 2
```text
可选机器学习模块
```

Cell 3
```text
事件质量评分或去噪
```

Cell 4
```text
低统计量增益与泛化表现
```

### Paragraph 62 · Heading 2

```text
5.2 低统计量设置
```

### Paragraph 63 · Body Text

```text
建议从同一批高统计量事件中随机抽取不同数量的子集，例如 1,000、5,000、10,000 和 50,000 条事件。每个统计量水平应重复多次，以避免单次随机抽样带来的偶然性。
```

### Paragraph 64 · Heading 2

```text
5.3 评价指标
```

### Paragraph 65 · List Bullet

```text
定位误差：重建高散射区域中心与真实目标中心之间的距离。
```

### Paragraph 66 · List Bullet

```text
轮廓重叠度：将重建结果阈值化后，与真实 U 型掩膜计算 IoU。
```

### Paragraph 67 · List Bullet

```text
图像相似度：条件具备时计算 SSIM 或 MSE，并说明真值图的构建方法。
```

### Paragraph 68 · List Bullet

```text
目标背景对比与信噪比：比较目标区域和背景区域的体素散射强度分布。
```

### Paragraph 69 · List Bullet

```text
计算代价：记录处理时间、内存消耗或单位事件的重建时间。
```

### Paragraph 70 · Heading 1

```text
6 面向低统计量的改进方法
```

### Paragraph 71 · Heading 2

```text
6.1 散射角筛选
```

### Paragraph 72 · Body Text

```text
小散射角事件通常对高散射目标的定位贡献有限，却可能带来背景累积。可设置散射角下限，或对不同散射角赋予不同权重。阈值不能仅通过视觉效果确定，应在验证集或重复实验中用定量指标选择。
```

### Paragraph 73 · Body Text

```text
【待补充】填写候选阈值范围、阈值搜索方法，以及最终阈值的选择依据。
```

### Paragraph 74 · Heading 2

```text
6.2 自适应体素重建
```

### Paragraph 75 · Body Text

```text
先用较大体素进行全局扫描，定位高散射候选区域；再在候选区域内使用较小体素进行局部重建。该策略的目标是在有限事件下减少全局噪声，并将计算资源集中于可能包含目标的位置。
```

### Paragraph 76 · Body Text

```text
【待补充】明确粗体素尺寸、细体素尺寸、候选区阈值、局部区域扩展范围和停止条件。
```

### Paragraph 77 · Heading 2

```text
6.3 可选的机器学习辅助模块
```

### Paragraph 78 · Body Text

```text
若前述物理改进已获得稳定结果，可训练轻量模型对事件质量进行评分，输入包括散射角、上下游轨迹参数、POCA 坐标和探测器命中完整性等特征。模型输出用于筛除低质量事件，随后仍使用相同 POCA 流程重建。
```

### Paragraph 79 · Body Text

```text
训练集与测试集必须在目标位置、尺寸、材料、事件数或探测器误差等方面有明确隔离，避免模型只记忆固定的 U 型几何。
```

### Paragraph 80 · Heading 1

```text
7 实验结果与讨论
```

### Paragraph 81 · Heading 2

```text
7.1 基线结果
```

### Paragraph 82 · Body Text

```text
【待补充】插入理想入射与半自然入射下的基线图像，并报告事件数、完整径迹率、定位误差和计算时间。
```

### Paragraph 83 · Heading 2

```text
7.2 低统计量退化规律
```

### Paragraph 84 · Body Text

```text
【待补充】插入事件数与 IoU、定位误差、信噪比之间的曲线；说明传统 POCA 在何种统计量范围内开始失效。
```

### Paragraph 85 · Heading 2

```text
7.3 改进方法对比
```

### Table 3

#### Row 1

Cell 1
```text
方法
```

Cell 2
```text
事件数
```

Cell 3
```text
IoU
```

Cell 4
```text
定位误差
```

Cell 5
```text
信噪比
```

Cell 6
```text
时间
```

#### Row 2

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 3

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 4

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 5

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

### Paragraph 86 · Normal

```text
表 X  不同方法在低统计量条件下的重建性能比较
```

### Paragraph 87 · Heading 2

```text
7.4 讨论与局限性
```

### Paragraph 88 · List Bullet

```text
讨论缪子源简化、探测器理想化、材料模型和统计误差对结论的影响。
```

### Paragraph 89 · List Bullet

```text
说明 U 型目标是否足以代表真实应用，以及后续需要加入哪些形状、位置和材料组合。
```

### Paragraph 90 · List Bullet

```text
若引入机器学习，讨论训练数据与测试数据的分布差异和泛化边界。
```

### Paragraph 91 · Heading 1

```text
8 结论与展望
```

### Paragraph 92 · Body Text

```text
【待补充】用 3 至 5 点总结可量化结论：基线能力、低统计量瓶颈、改进方法增益、计算代价和适用范围。不要在结论中引入新数据。
```

### Paragraph 93 · Body Text

```text
【待补充】展望可包括：更真实的宇宙线缪子源、探测器分辨率与效率建模、更多目标场景、真实实验数据，以及物理约束机器学习方法。
```

### Paragraph 94 · Heading 1

```text
参考文献
```

### Paragraph 95 · Body Text

```text
【待补充】使用学校或目标期刊要求的格式统一整理。建议使用 Zotero 管理文献，并在正文中及时插入引用占位符，避免最后集中补文献。
```

### Paragraph 97 · Heading 1

```text
附录 A 实验记录表
```

### Paragraph 98 · Body Text

```text
每次运行建议记录以下信息。这个表可直接复制追加，用于保证论文图表的可复现性。
```

### Table 4

#### Row 1

Cell 1
```text
日期
```

Cell 2
```text
代码版本
```

Cell 3
```text
入射模式
```

Cell 4
```text
事件数
```

Cell 5
```text
方法与参数
```

Cell 6
```text
输出图或 CSV
```

#### Row 2

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 3

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 4

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 5

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 6

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```

#### Row 7

Cell 1
```text
【填写】
```

Cell 2
```text
【填写】
```

Cell 3
```text
【填写】
```

Cell 4
```text
【填写】
```

Cell 5
```text
【填写】
```

Cell 6
```text
【填写】
```
