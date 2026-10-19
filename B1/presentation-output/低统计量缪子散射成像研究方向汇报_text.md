# Slide text extraction: 低统计量缪子散射成像研究方向汇报.pptx

Source: `B1/presentation-output/低统计量缪子散射成像研究方向汇报.pptx`

Source SHA-256: `2e1642321e4ac57d25f3d94682d43232f23b8e872208c6b16d8e0c2d96c79b60`

Slides: 6

Extraction: `python-pptx 1.0.2`; text objects in slide shape order, followed by speaker notes when present. The original PPTX remains authoritative for layout, graphics, and image-embedded text.

## Slide 1

### Text object 1

```text
面向低统计量条件的物理约束自适应缪子散射成像方法
```

### Text object 2

```text
以传统 POCA 为基线，研究更少缪子事件下的轮廓恢复
```

### Text object 3

```text
核心问题
```

### Text object 4

```text
低统计量时，PoCA 图像的 U 型轮廓容易破碎，目标定位不稳定
```

### Text object 5

```text
拟解决方向：散射角筛选 + 自适应体素重建
```

### Text object 6

```text
研究生阶段汇报
```

### Speaker notes

```text
汇报主线：先证明确实存在低统计量问题，再说明用两个可解释的物理改进方法解决它。
```

## Slide 2

### Text object 1

```text
研究问题与创新点
```

### Text object 2

```text
传统方法的限制
```

### Text object 3

```text
固定体素 PoCA
```

### Text object 4

```text
少量事件分散到整个成像区域
背景累积后，轮廓与空腔难以稳定恢复
```

### Text object 5

```text
本研究的创新
```

### Text object 6

```text
把“低统计量”作为核心实验条件
```

### Text object 7

```text
在同一套 Geant4 几何、同一批事件和同一评价指标下，
比较传统 POCA 与两种物理可解释的改进：
```

### Text object 8

```text
1  散射角筛选：减少低信息量事件造成的背景累积
```

### Text object 9

```text
2  自适应体素：把有限事件集中用于候选高散射区域
```

### Text object 10

```text
评价标准：同事件数下 IoU 更高，定位误差更小
```

### Speaker notes

```text
创新不是简单换网络，而是在固定物理基线中系统处理低统计量问题。后续机器学习只作为可选辅助。
```

## Slide 3

### Text object 1

```text
已完成的 PoCA 基线
```

### Text object 2

```text
完整事件数下，U 型轮廓已经可以恢复
```

### Text object 3

```text
9,975
```

### Text object 4

```text
完整四层径迹数
```

### Text object 5

```text
0.8409
```

### Text object 6

```text
等面积 IoU
```

### Text object 7

```text
3.80 mm
```

### Text object 8

```text
定位误差
```

### Text object 9

```text
结论：几何、CSV 导出、PoCA 与评价链路均已跑通。
这张图是后续改进方法的统一对照基线。
```

### Speaker notes

```text
数据来源：muon_analysis/baseline_results/poca_xy_9975_events.png。固定 32×32×32 体素，theta² 累积，理想 4 GeV 平行入射。
```

## Slide 4

### Text object 1

```text
低统计量是当前基线的主要瓶颈
```

### Text object 2

```text
低统计量证据
```

### Text object 3

```text
1,000 条事件
```

### Text object 4

```text
IoU 0.2836
定位误差 9.46 mm
```

### Text object 5

```text
5,000 条事件
```

### Text object 6

```text
IoU 0.5916
定位误差 4.36 mm
```

### Text object 7

```text
研究动机：降低获得可用图像所需的事件数
```

### Speaker notes

```text
数据来源：muon_analysis/baseline_results/baseline_metrics_summary.csv。1000、3000、5000 条事件各随机无放回抽样 5 次；9975 条使用全部样本。
```

## Slide 5

### Text object 1

```text
拟研究的方法
```

### Text object 2

```text
所有方法使用相同的 Geant4 数据、目标几何和评价指标
```

### Text object 3

```text
传统固定体素 PoCA
```

### Text object 4

```text
基线：全部事件进入均匀网格
```

### Text object 5

```text
散射角筛选 PoCA
```

### Text object 6

```text
扫描阈值，减少小散射角事件
```

### Text object 7

```text
自适应体素 PoCA
```

### Text object 8

```text
粗定位后，对候选区域细化网格
```

### Text object 9

```text
验证问题：在 1,000 到 5,000 条事件下，能否提高 IoU 并降低定位误差？
```

### Speaker notes

```text
研究顺序：先完成角度筛选，再完成自适应体素。机器学习不作为当前论文主线。
```

## Slide 6

### Text object 1

```text
实验设计与预期论文贡献
```

### Text object 2

```text
实验设计
```

### Text object 3

```text
• U 型铝目标，四层探测器，理想入射
• 事件数：1,000、3,000、5,000、9,975
• 指标：IoU、定位误差、运行时间
• 每个低统计量组重复抽样 5 次
```

### Text object 4

```text
论文贡献
```

### Text object 5

```text
• 给出传统 POCA 在低统计量下的定量退化基线
• 提出散射角筛选与自适应体素的组合策略
• 用同条件仿真证明改进是否减少所需事件数
```

### Speaker notes

```text
当前已完成 PoCA 基线。下一阶段产出：角度阈值扫描的对照表和图，再进入自适应体素。
```
