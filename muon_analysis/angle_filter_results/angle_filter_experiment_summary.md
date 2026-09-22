# 散射角筛选 PoCA 实验

## 实验目的

在固定入射事件数量下扫描最小散射角阈值，验证筛除小角散射事件能否改善低统计量 PoCA 成像。

## 公平比较原则

每次随机抽样得到的一批入射事件，会被全部阈值共同使用。阈值只改变保留事件，不改变采集事件数。

## 输出文件

- `angle_filter_metrics_by_repeat.csv`：每次抽样、每个阈值的原始指标。
- `angle_filter_metrics_summary.csv`：均值与标准差。
- `angle_filter_metrics_vs_threshold.png`：IoU、定位误差、保留比例随阈值变化。
- `angle_filter_poca_xy_3000_events.png`：3000 个入射事件的阈值扫描成像图。

## 解释规则

只有当某个阈值在多个低统计量水平下稳定提高 IoU 或降低定位误差，且保留事件比例合理时，才能把它称为有效改进。
