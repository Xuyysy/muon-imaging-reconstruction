# Slide text extraction: muon-imaging-research-direction-v2.pptx

Source: `B1/presentation-output/muon-imaging-research-direction-v2.pptx`

Source SHA-256: `b29628296a2333fcf0628eff26f3599da8f0f218c57001455483aabcbd30c369`

Slides: 11

Extraction: `python-pptx 1.0.2`; text objects in slide shape order, followed by speaker notes when present. The original PPTX remains authoritative for layout, graphics, and image-embedded text.

## Slide 1

### Text object 1

```text
研究方向汇报
```

### Text object 2

```text
面向低统计量条件的
物理约束自适应缪子散射成像
```

### Text object 3

```text
从 POCA 基线出发，研究自适应体素与可靠事件筛选
```

### Text object 4

```text
Geant4 仿真 · Python 分析 · 低统计量重建
```

### Text object 5

```text
汇报人：徐耀阳
用于与导师讨论课题可行性
```

### Text object 6

```text
目标物
```

### Text object 7

```text
四层探测器与散射目标示意
```

### Speaker notes

```text
本页为研究方向汇报封面。项目当前基线为四层 Geant4 探测器、POCA 重建和 Python 固定体素分析。
```

## Slide 2

### Text object 1

```text
01 研究背景
```

### Text object 2

```text
低统计量是缪子成像落地时的核心约束
```

### Text object 3

```text
02
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
自然缪子通量有限，短时间采集会让体素统计不足。POCA 又把每条事件压缩为单个散射点，因此噪声和定位偏差会被放大。
```

### Text object 6

```text
物理约束
```

### Text object 7

```text
算法瓶颈
```

### Text object 8

```text
研究机会
```

### Text object 9

```text
缪子数量有限
轨迹存在测量误差
散射角具有长尾
```

### Text object 10

```text
固定体素不适合所有统计量
POCA 空间信息利用不足
大角度事件制造伪影
```

### Text object 11

```text
根据局部统计量调节重建
根据事件质量分配权重
保留可解释的物理基线
```

### Text object 12

```text
研究判断：问题具有应用价值，但“大体素 + 角度截断”已有先例，创新必须放在自适应机制与跨场景验证上。
```

### Speaker notes

```text
来源：Multiparametric approach to the assessment of muon tomographic results, EPJ Plus, https://link.springer.com/article/10.1140/epjp/s13360-020-00970-z；A new efficient imaging reconstruction method for muon scattering tomography, NIMA 2024, https://www.sciencedirect.com/science/article/pii/S0168900224008581。
```

## Slide 3

### Text object 1

```text
02 已有基础
```

### Text object 2

```text
当前项目已经形成可复现的 POCA 基线
```

### Text object 3

```text
03
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
四层探测器
```

### Text object 6

```text
轨迹拟合
```

### Text object 7

```text
散射角计算
```

### Text object 8

```text
POCA 坐标
```

### Text object 9

```text
固定体素图像
```

### Text object 10

```text
代码与数据位置
```

### Text object 11

```text
B1：Geant4 四层探测器和运行宏
```

### Text object 12

```text
muon_analysis：CSV 输出与 analysis.ipynb
```

### Text object 13

```text
GitHub：私有仓库，基线标签 v0.1-poca-baseline
```

### Text object 14

```text
当前基线能回答的问题
```

### Text object 15

```text
在 1,000、3,000、10,000 个事件下，POCA 图像会怎样变化？
```

## Slide 4

### Text object 1

```text
03 文献与方法
```

### Text object 2

```text
已有方法证明问题重要，也限定了我们的创新边界
```

### Text object 3

```text
04
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
相关工作已经覆盖 POCA 改进、统计重建和机器学习增强。我们的切入点应放在“自动适应低统计量”，而不是重复已有网络结构。
```

### Text object 6

```text
方法
```

### Text object 7

```text
优势
```

### Text object 8

```text
局限
```

### Text object 9

```text
本课题中的作用
```

### Text object 10

```text
POCA
```

### Text object 11

```text
简单、快速、易复现
```

### Text object 12

```text
空间信息利用不足，对低统计量敏感
```

### Text object 13

```text
当前基线
```

### Text object 14

```text
MLEM / ASR
```

### Text object 15

```text
统计模型更完整
```

### Text object 16

```text
实现和计算成本较高
```

### Text object 17

```text
文献对照或后续扩展
```

### Text object 18

```text
大体素 + 角度截断
```

### Text object 19

```text
降低统计涨落和长尾伪影
```

### Text object 20

```text
参数通常固定，已有研究基础
```

### Text object 21

```text
重要基线，不能直接当作全部创新
```

### Text object 22

```text
CNN / U-Net
```

### Text object 23

```text
能改善图像外观和噪声
```

### Text object 24

```text
依赖数据与泛化验证
```

### Text object 25

```text
后续扩展，不作为第一步
```

### Text object 26

```text
研究空白：体素大小、事件权重和角度阈值能否根据局部统计量自动调整，并在跨场景条件下保持稳定？
```

### Speaker notes

```text
文献来源：POCA 局限与应用：https://link.springer.com/article/10.1140/epjp/s13360-020-00970-z；大体素与角度截断：https://www.sciencedirect.com/science/article/pii/S0168900224008581；机器学习图像增强：https://www.sciencedirect.com/science/article/pii/S0168900224006818；U-Net 短时间成像增强：https://doi.org/10.1016/j.nima.2026.005231。
```

## Slide 5

### Text object 1

```text
04 研究问题
```

### Text object 2

```text
研究目标从“能成像”推进到“少事件也能可靠成像”
```

### Text object 3

```text
05
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
核心问题
```

### Text object 6

```text
在事件数量有限、轨迹存在误差时，如何利用物理特征自适应调整重建策略？
```

### Text object 7

```text
三个可验证假设
```

### Text object 8

```text
1
```

### Text object 9

```text
自适应体素能减少低统计量下的空体素和噪声
```

### Text object 10

```text
2
```

### Text object 11

```text
事件质量评分能降低异常大散射角事件造成的伪影
```

### Text object 12

```text
3
```

### Text object 13

```text
改进方法在目标位置、尺寸和误差变化后仍保持稳定
```

## Slide 6

### Text object 1

```text
05 方法框架
```

### Text object 2

```text
物理模型负责解释，数据驱动模块负责筛选与调节
```

### Text object 3

```text
06
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
Geant4
事件
```

### Text object 6

```text
轨迹与
散射角
```

### Text object 7

```text
事件质量
评分
```

### Text object 8

```text
自适应
体素重建
```

### Text object 9

```text
图像与
指标
```

### Text object 10

```text
输入：击中位置、轨迹方向、散射角、POCA 坐标
```

### Text object 11

```text
输出：三维体素图、目标位置、图像质量、所需事件数
```

### Text object 12

```text
权重示意
```

### Text object 13

```text
wᵢ = qᵢ × min(θᵢ², θₘₐₓ²)
```

### Speaker notes

```text
权重公式是研究起点，不是已验证结论。q_i 表示事件质量评分，后续需要通过仿真和消融实验确定定义。
```

## Slide 7

### Text object 1

```text
06 计划创新
```

### Text object 2

```text
创新点集中在自适应机制与可靠性验证
```

### Text object 3

```text
07
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
自适应体素
```

### Text object 6

```text
根据局部事件密度和统计量自动调整体素大小，避免固定网格在低统计量下失效。
```

### Text object 7

```text
物理事件筛选
```

### Text object 8

```text
用散射角、轨迹质量、POCA 几何特征筛选或加权事件，保留物理可解释性。
```

### Text object 9

```text
跨场景泛化
```

### Text object 10

```text
改变目标位置、尺寸、材料和探测器误差，验证方法是否真正学到成像规律。
```

### Text object 11

```text
论文贡献的最低要求：改进方法必须在相同事件数下优于固定体素 POCA，并通过多场景测试。
```

### Speaker notes

```text
相关研究已经使用大体素和角度截断改善 POCA，因此本课题不能把它们本身作为全部创新。相关研究：https://www.sciencedirect.com/science/article/pii/S0168900224008581。机器学习图像增强已有研究：https://www.sciencedirect.com/science/article/pii/S0168900224006818。
```

## Slide 8

### Text object 1

```text
07 实验设计
```

### Text object 2

```text
先建立低统计量证据，再验证改进方法
```

### Text object 3

```text
08
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
变量
```

### Text object 6

```text
设置
```

### Text object 7

```text
事件数
```

### Text object 8

```text
500 / 1,000 / 3,000 / 10,000
```

### Text object 9

```text
目标物
```

### Text object 10

```text
位置、尺寸、材料、单目标与多目标
```

### Text object 11

```text
误差
```

### Text object 12

```text
击中位置误差、探测器位置误差、入射角变化
```

### Text object 13

```text
方法
```

### Text object 14

```text
固定体素 POCA / 角度截断 / 自适应体素 / 事件筛选
```

### Text object 15

```text
评价指标
```

### Text object 16

```text
MSE · SSIM · IoU · 质心定位误差 · 图像信噪比 · 目标检测率 · 运行时间
```

## Slide 9

### Text object 1

```text
08 实施计划
```

### Text object 2

```text
四个版本对应四个可交付研究阶段
```

### Text object 3

```text
09
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
v0.1
```

### Text object 6

```text
POCA 基线
```

### Text object 7

```text
已完成
```

### Text object 8

```text
v0.2
```

### Text object 9

```text
低统计量分析
```

### Text object 10

```text
下一步
```

### Text object 11

```text
v0.3
```

### Text object 12

```text
自适应体素
```

### Text object 13

```text
核心算法
```

### Text object 14

```text
v0.4
```

### Text object 15

```text
物理事件筛选
```

### Text object 16

```text
扩展验证
```

### Text object 17

```text
每个阶段的完成标准
```

### Text object 18

```text
代码可运行，实验参数可记录
```

### Text object 19

```text
结果有定量指标和基线对比
```

### Text object 20

```text
完成后提交 Git 并创建版本标签
```

## Slide 10

### Text object 1

```text
09 风险与边界
```

### Text object 2

```text
先控制研究风险，再扩大模型复杂度
```

### Text object 3

```text
10
```

### Text object 4

```text
研究方向讨论稿
```

### Text object 5

```text
仿真与真实探测器差距
```

### Text object 6

```text
加入击中位置误差、探测器偏移和不同入射角；保留真实数据验证计划
```

### Text object 7

```text
创新点与已有工作接近
```

### Text object 8

```text
把重点放在自动自适应机制、事件质量评分和跨场景泛化
```

### Text object 9

```text
模型记住目标模板
```

### Text object 10

```text
测试集改变位置、尺寸、材料和事件数，报告跨场景性能
```

### Text object 11

```text
评价指标不一致
```

### Text object 12

```text
固定数据划分、固定体素范围，并同时报告图像与定位指标
```

## Slide 11

### Text object 1

```text
导师讨论
```

### Text object 2

```text
这个课题是否值得
作为硕士阶段主线？
```

### Text object 3

```text
希望重点确认三件事
```

### Text object 4

```text
低统计量 + 自适应体素是否具备足够创新性？
```

### Text object 5

```text
是否需要尽早加入真实探测器数据？
```

### Text object 6

```text
优先投算法类期刊，还是核技术应用类期刊？
```

### Text object 7

```text
建议的近期工作
```

### Text object 8

```text
先完成 N = 500 / 1,000 / 3,000 / 10,000 的 POCA 对比，再决定自适应体素的具体形式。
```

### Text object 9

```text
谢谢老师指导
```
