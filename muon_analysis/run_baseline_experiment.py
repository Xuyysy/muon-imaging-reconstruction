"""生成可复现的低统计量固定体素 PoCA 基线实验。

这个脚本是论文中“传统 POCA 对照组”的一键实验版本：

1. 读取 Geant4 输出的原始 ``muon_tracks.csv``；
2. 对 1,000、3,000、5,000 和全部事件分别进行固定体素重建；
3. 计算 IoU、定位误差、目标/背景对比和运行时间；
4. 将图像、CSV 指标和实验配置写到 ``baseline_results/``。

重要：脚本只读取原始 CSV，绝不会覆盖或修改 Geant4 生成的数据。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# 项目根目录。这样无论从哪个终端目录运行，输入和输出位置都固定。
ROOT = Path('/Users/xuyaoyang/geant4-test')
# 原始事件：由 Geant4 四层探测器模拟写出。
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
# 派生结果：论文图表和统计指标统一放在这里，保持原始数据干净。
OUT_DIR = ROOT / 'muon_analysis' / 'baseline_results'

# 低统计量对照组。全部事件会在 main() 中自动追加进去。
EVENT_COUNTS = [1000, 3000, 5000]
# 小样本每个事件数重复五次，降低“刚好抽到一批好事件”的偶然影响。
REPEATS = 5
# 固定种子让每次运行得到同一批子样本，论文结果可以复现。
RANDOM_SEED = 20260921
# 三维规则网格：x、y、z 各划分为 32 个体素。
BINS = 32
# 成像区域半宽。U 型目标位于约 ±60 mm 内，额外留出背景区域。
XY_LIMIT_MM = 90.0
Z_LIMIT_MM = 100.0


def voxelize_poca(data: pd.DataFrame):
    """把一批 PoCA 点累积到固定三维体素。

    每个体素的强度为其中所有事件 ``theta_mrad²`` 的和。
    角度平方会让明显散射的缪子比近似直行的缪子拥有更大贡献。

    返回：
        volume: shape=(BINS, BINS, BINS) 的三维强度数组；
        edges: 三个坐标方向的体素边界；
        valid: 真正进入成像区域的事件，用于记录统计量。
    """
    # 范围外的 PoCA 点不参与成像，但输入 data 本身不会被改动。
    valid = data[
        (data['poca_x_mm'].abs() <= XY_LIMIT_MM)
        & (data['poca_y_mm'].abs() <= XY_LIMIT_MM)
        & (data['poca_z_mm'].abs() <= Z_LIMIT_MM)
    ]
    # np.histogramdd 使用的是“边界”，因此每个方向要 BINS + 1 个数值。
    edges = [
        np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1),
        np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1),
        np.linspace(-Z_LIMIT_MM, Z_LIMIT_MM, BINS + 1),
    ]
    # 把 (x, y, z) 坐标落到三维网格，并按散射角平方加权累积。
    volume, edges = np.histogramdd(
        valid[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy(),
        bins=edges,
        weights=valid['theta_mrad'].to_numpy() ** 2,
    )
    return volume, edges, valid


def u_target_mask(x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    """Ground-truth XY mask from DetectorConstruction.cc, in millimetres.

    U shape: 120 mm x 120 mm footprint, 30 mm wall thickness, opening to +y.
    """
    # 用体素中心生成二维坐标网格；mask 的坐标和重建图像严格对齐。
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    # 三部分与 DetectorConstruction.cc 中的三个铝盒子一一对应。
    bottom = (np.abs(x_grid) <= 60.0) & (y_grid >= -60.0) & (y_grid <= -30.0)
    left = (x_grid >= -60.0) & (x_grid <= -30.0) & (y_grid >= -30.0) & (y_grid <= 60.0)
    right = (x_grid >= 30.0) & (x_grid <= 60.0) & (y_grid >= -30.0) & (y_grid <= 60.0)
    return bottom | left | right


def evaluate_image(image: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray):
    """评价一张 xy 投影图的轮廓恢复和定位精度。

    IoU 采用等面积 Top-k 阈值：预测图中强度最高的像素数，固定为
    U 型真值掩膜的像素数。这样不同事件数不需要各自手动调阈值。
    """
    truth = u_target_mask(x_centers, y_centers)
    # 真值区域像素数决定预测区域面积，保证 IoU 比较公平。
    k = int(truth.sum())
    predicted = np.zeros_like(truth, dtype=bool)
    top_indices = np.argpartition(image.ravel(), -k)[-k:]
    predicted.ravel()[top_indices] = True
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    iou = intersection / union if union else np.nan

    # 用散射强度加权质心表示“重建目标中心”，再和真值中心比较。
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    total_weight = float(image.sum())
    if total_weight > 0:
        rec_x = float((image * x_grid).sum() / total_weight)
        rec_y = float((image * y_grid).sum() / total_weight)
        true_x = float(x_grid[truth].mean())
        true_y = float(y_grid[truth].mean())
        localization_error = float(np.hypot(rec_x - true_x, rec_y - true_y))
    else:
        localization_error = np.nan

    # 目标/背景对比仅作为辅助观察量，不作为主结论：它对局部热点较敏感。
    target_mean = float(image[truth].mean())
    background_mean = float(image[~truth].mean())
    contrast_ratio = target_mean / background_mean if background_mean > 0 else np.nan
    return iou, localization_error, contrast_ratio, int(truth.sum())


FONT_PATH = '/Library/Fonts/Arial Unicode.ttf'


def font(size, bold=False):
    """优先使用 macOS 的 Unicode 字体；找不到时退回默认字体。"""
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except OSError:
        return ImageFont.load_default()


def inferno_like(values):
    """无需 matplotlib 的简化 inferno 色图，供 Pillow 输出 PNG 使用。"""
    stops = np.array([
        [0.00, 0, 0, 4], [0.20, 45, 11, 86], [0.43, 132, 32, 107],
        [0.66, 211, 72, 72], [0.83, 248, 149, 64], [1.00, 252, 255, 164],
    ], dtype=float)
    flat = np.clip(values.ravel(), 0, 1)
    rgb = np.empty((flat.size, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[:, channel] = np.interp(flat, stops[:, 0], stops[:, channel + 1]).astype(np.uint8)
    return rgb.reshape((*values.shape, 3))


def plot_image(image, edges, count, used_count, output_path):
    """绘制单个事件数下的 PoCA xy 投影图并保存 PNG。"""
    canvas = Image.new('RGB', (1050, 820), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, tick_font = font(25), font(20), font(16)
    left, top, size = 125, 120, 620
    draw.text((left, 28), f'Fixed-voxel PoCA baseline: {count} events', fill='black', font=title_font)
    draw.text((left, 62), f'{used_count} PoCA points in imaging volume', fill='#444444', font=label_font)
    # 体素强度跨度较大，log(1 + intensity) 让背景和目标边缘都可见。
    scaled = np.log1p(image)
    vmax = float(scaled.max())
    normalised = scaled / vmax if vmax > 0 else scaled
    rgb = inferno_like(normalised.T[::-1, :])
    heatmap = Image.fromarray(rgb, 'RGB').resize((size, size), Image.Resampling.NEAREST)
    canvas.paste(heatmap, (left, top))
    draw.rectangle((left, top, left + size, top + size), outline='black', width=2)
    for value in [-90, -45, 0, 45, 90]:
        px = left + round((value + 90) / 180 * size)
        py = top + size - round((value + 90) / 180 * size)
        draw.line((px, top + size, px, top + size + 7), fill='black', width=1)
        draw.text((px - 14, top + size + 12), str(value), fill='black', font=tick_font)
        draw.line((left - 7, py, left, py), fill='black', width=1)
        draw.text((left - 54, py - 9), str(value), fill='black', font=tick_font)
    draw.text((left + size // 2 - 50, top + size + 50), 'PoCA x [mm]', fill='black', font=label_font)
    draw.text((20, top + size // 2 - 12), 'PoCA y [mm]', fill='black', font=label_font)
    bar_x, bar_y, bar_w, bar_h = 805, top, 32, size
    gradient = np.linspace(1, 0, bar_h).reshape((bar_h, 1))
    bar = Image.fromarray(inferno_like(gradient).repeat(bar_w, axis=1), 'RGB')
    canvas.paste(bar, (bar_x, bar_y))
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline='black', width=1)
    draw.text((770, bar_y - 33), 'log(1 + sum theta^2)', fill='black', font=tick_font)
    draw.text((846, bar_y), f'{vmax:.2f}', fill='black', font=tick_font)
    draw.text((846, bar_y + bar_h - 18), '0', fill='black', font=tick_font)
    canvas.save(output_path)


def plot_metrics(summary: pd.DataFrame, output_path: Path):
    """将 IoU、定位误差和目标/背景对比随事件数的变化画在一张图中。"""
    metrics = [
        ('iou_mean', 'iou_std', 'Equal-area IoU'),
        ('localization_error_mm_mean', 'localization_error_mm_std', 'Localization error [mm]'),
        ('contrast_ratio_mean', 'contrast_ratio_std', 'Target/background contrast'),
    ]
    canvas = Image.new('RGB', (1500, 560), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 22), 'Low-statistics fixed-voxel PoCA baseline (mean +/- standard deviation)', fill='black', font=font(25))
    counts = summary['event_count'].to_numpy(dtype=float)
    x_min, x_max = np.log10(counts.min()), np.log10(counts.max())
    for panel, (mean_col, std_col, label) in enumerate(metrics):
        left, top, width, height = 70 + panel * 480, 100, 365, 325
        values = summary[mean_col].to_numpy(dtype=float)
        errors = summary[std_col].fillna(0).to_numpy(dtype=float)
        lo = float(np.min(values - errors))
        hi = float(np.max(values + errors))
        pad = max((hi - lo) * 0.15, 1e-6)
        lo, hi = lo - pad, hi + pad
        draw.rectangle((left, top, left + width, top + height), outline='black', width=2)
        draw.text((left, top - 34), label, fill='black', font=font(18))
        points = []
        for x, value, error in zip(counts, values, errors):
            px = left + (np.log10(x) - x_min) / (x_max - x_min) * width if x_max > x_min else left + width / 2
            py = top + height - (value - lo) / (hi - lo) * height
            e_top = top + height - (value + error - lo) / (hi - lo) * height
            e_bottom = top + height - (value - error - lo) / (hi - lo) * height
            draw.line((px, e_top, px, e_bottom), fill='#2A5C9B', width=2)
            draw.line((px - 5, e_top, px + 5, e_top), fill='#2A5C9B', width=2)
            draw.line((px - 5, e_bottom, px + 5, e_bottom), fill='#2A5C9B', width=2)
            points.append((px, py))
            draw.text((px - 19, top + height + 10), f'{int(x)}', fill='black', font=font(14))
        if len(points) > 1:
            draw.line(points, fill='#2A5C9B', width=3)
        for px, py in points:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill='#2A5C9B')
        draw.text((left + 115, top + height + 42), 'Input events (log scale)', fill='black', font=font(15))
        draw.text((left - 58, top + 2), f'{hi:.3g}', fill='black', font=font(13))
        draw.text((left - 58, top + height - 16), f'{lo:.3g}', fill='black', font=font(13))
    canvas.save(output_path)


def main():
    # 创建结果文件夹；若已存在则保留文件夹并覆盖同名派生结果。
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 原始 CSV 只读入内存，不在这里写回。
    data = pd.read_csv(INPUT_CSV)
    required = {'poca_x_mm', 'poca_y_mm', 'poca_z_mm', 'theta_mrad'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    # 过滤掉大于现有数据量的请求，并自动把“全部事件”加入最后一组。
    counts = [count for count in EVENT_COUNTS if count <= len(data)] + [len(data)]
    counts = list(dict.fromkeys(counts))
    rng = np.random.default_rng(RANDOM_SEED)
    # 评价指标需要体素中心坐标，因此先由边界计算中心。
    x_edges = np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1)
    y_edges = np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    records = []
    for count in counts:
        # 全部事件只有一种组合；低统计量子集要重复随机抽样。
        repeat_count = 1 if count == len(data) else REPEATS
        representative = None
        for repeat in range(repeat_count):
            if count == len(data):
                subset = data
            else:
                # 无放回抽样：同一次实验内不重复使用同一个缪子事件。
                indices = rng.choice(len(data), size=count, replace=False)
                subset = data.iloc[indices]
            # 只计体素化所需时间，便于以后与改进方法做同口径比较。
            start = time.perf_counter()
            volume, edges, used = voxelize_poca(subset)
            image = volume.sum(axis=2)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            iou, localization_error, contrast_ratio, truth_pixels = evaluate_image(
                image, x_centers, y_centers
            )
            records.append({
                'event_count': count,
                'repeat': repeat + 1,
                'input_events': len(subset),
                'poca_points_in_volume': len(used),
                'iou_equal_area': iou,
                'localization_error_mm': localization_error,
                'target_background_contrast': contrast_ratio,
                'voxelization_runtime_ms': runtime_ms,
                'truth_u_pixels': truth_pixels,
            })
            # 每个事件数保存第 1 次抽样的图，定量结论仍以全部重复的均值/标准差为准。
            if repeat == 0:
                representative = (image, edges, len(used))
        image, edges, used_count = representative
        plot_image(image, edges, count, used_count, OUT_DIR / f'poca_xy_{count}_events.png')

    # 原始指标保留每一次重复，汇总指标用于论文中的表格和曲线。
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT_DIR / 'baseline_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby('event_count', as_index=False).agg(
        repeats=('repeat', 'count'),
        input_events=('input_events', 'first'),
        poca_points_in_volume_mean=('poca_points_in_volume', 'mean'),
        poca_points_in_volume_std=('poca_points_in_volume', 'std'),
        iou_mean=('iou_equal_area', 'mean'),
        iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
        contrast_ratio_mean=('target_background_contrast', 'mean'),
        contrast_ratio_std=('target_background_contrast', 'std'),
        runtime_ms_mean=('voxelization_runtime_ms', 'mean'),
        runtime_ms_std=('voxelization_runtime_ms', 'std'),
    )
    summary.to_csv(OUT_DIR / 'baseline_metrics_summary.csv', index=False)
    plot_metrics(summary, OUT_DIR / 'baseline_metrics_vs_event_count.png')

    # 将所有关键参数写入 JSON，避免日后只剩图片却不知道如何复现。
    config = {
        'source_csv': str(INPUT_CSV),
        'input_rows': int(len(data)),
        'source_mode': 'Ideal (existing baseline CSV)',
        'random_seed': RANDOM_SEED,
        'event_counts': counts,
        'repeats': REPEATS,
        'voxel_grid': [BINS, BINS, BINS],
        'xy_limit_mm': XY_LIMIT_MM,
        'z_limit_mm': Z_LIMIT_MM,
        'weight': 'sum(theta_mrad^2)',
        'iou_definition': 'Top-k prediction with k equal to the U-shaped ground-truth pixel count.',
        'target_geometry': 'U-shaped aluminium target: 120 mm x 120 mm XY footprint, 30 mm wall thickness, opening toward +y.',
    }
    (OUT_DIR / 'baseline_experiment_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )

    # 生成一份便于直接阅读的 Markdown 实验说明，标注论文中应引用的章节。
    lines = [
        '# 固定体素 POCA 低统计量基线实验',
        '',
        '## 实验设置',
        '',
        f'- 输入：`muon_tracks.csv`，共 {len(data)} 条完整径迹。',
        '- 缪子源：Ideal 模式，固定 4 GeV、平行入射。',
        f'- 成像区域：x、y 为 ±{XY_LIMIT_MM:g} mm；z 为 ±{Z_LIMIT_MM:g} mm；体素网格为 {BINS}×{BINS}×{BINS}。',
        '- 体素强度：该体素内所有事件的 `theta_mrad²` 之和。',
        f'- 子样本：1000、3000、5000 条事件各进行 {REPEATS} 次无放回随机抽样；完整样本仅计算 1 次。',
        '- IoU：使用与 U 型真值掩膜像素数相同的 Top-k 重建像素计算，避免为每组事件数分别调阈值。',
        '',
        '## 输出文件',
        '',
        '- `poca_xy_*_events.png`：四个统计量水平的代表性 XY PoCA 图。',
        '- `baseline_metrics_by_repeat.csv`：每次随机抽样的原始指标。',
        '- `baseline_metrics_summary.csv`：均值和标准差汇总。',
        '- `baseline_metrics_vs_event_count.png`：指标随事件数变化的图。',
        '',
        '## 论文草稿对应位置',
        '',
        '- 第 4.3 节：固定体素 POCA 实现和代表性图像。',
        '- 第 5.2 节：低统计量设置。',
        '- 第 5.3 节：IoU、定位误差、目标背景对比和运行时间的定义。',
        '- 第 7.1 节：基线图像与完整事件数结果。',
        '- 第 7.2 节：事件数减少导致的成像退化规律。',
    ]
    (OUT_DIR / 'baseline_experiment_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
