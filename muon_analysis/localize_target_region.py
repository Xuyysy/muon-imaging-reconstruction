"""仅根据 PoCA 图像估计目标的局部位置，不读取目标真值。

固定规则：对 xy 投影做 3×3 平均平滑，取强度最高的 15% 图像像素，
在八邻域中寻找总强度最大的连通区域，最后计算该区域的强度加权中心。
15% 是预先固定的粗定位预算；实验中不针对场景重新调节。
"""
from __future__ import annotations

import numpy as np

TOP_PIXEL_FRACTION = .15
SMOOTHING_WIDTH = 3
WINDOW_BINS = 24
# 三种候选尺度相当于约 79、101、135 mm。它们仅由成像网格定义，
# 不会根据目标真值、目标类别或验证集定位误差逐个调整。
MULTISCALE_WINDOW_BINS = (14, 18, 24)
ROBUST_POSITIVE_QUANTILE = .95


def smooth_3x3(image: np.ndarray) -> np.ndarray:
    """计算零填充边界的 3×3 均值，轻微连接相邻散射热点。"""
    padded = np.pad(image, 1, mode='constant')
    return sum(padded[dx:dx + image.shape[0], dy:dy + image.shape[1]]
               for dx in range(3) for dy in range(3)) / 9.0


def localize_target_region(image: np.ndarray, x_centers: np.ndarray,
                           y_centers: np.ndarray) -> dict:
    """返回最大高散射连通区域的坐标、大小和边界标记。

    这里的输入只有重建图和体素坐标，没有目标位置、尺寸或材料真值。
    """
    if image.ndim != 2 or image.shape != (len(x_centers), len(y_centers)):
        raise ValueError('image shape must match x/y bin centers')
    if np.any(image < 0) or not np.isfinite(image).all():
        raise ValueError('image must contain finite nonnegative intensities')
    if not np.any(image > 0):
        return {'x_mm': np.nan, 'y_mm': np.nan, 'selected_pixels': 0,
                'component_pixels': 0, 'touches_boundary': False}

    smoothed = smooth_3x3(image)
    n_top = max(1, int(np.ceil(image.size * TOP_PIXEL_FRACTION)))
    selected = np.zeros(image.size, dtype=bool)
    order = np.argsort(-smoothed.ravel(), kind='stable')[:n_top]
    selected[order] = smoothed.ravel()[order] > 0
    selected = selected.reshape(image.shape)

    visited = np.zeros_like(selected)
    best_pixels: list[tuple[int, int]] = []
    best_strength = -1.0
    nx, ny = image.shape
    for start_x, start_y in np.argwhere(selected):
        if visited[start_x, start_y]:
            continue
        stack = [(int(start_x), int(start_y))]
        visited[start_x, start_y] = True
        component = []
        strength = 0.0
        while stack:
            ix, iy = stack.pop()
            component.append((ix, iy))
            strength += float(smoothed[ix, iy])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    jx, jy = ix + dx, iy + dy
                    if (0 <= jx < nx and 0 <= jy < ny and selected[jx, jy]
                            and not visited[jx, jy]):
                        visited[jx, jy] = True
                        stack.append((jx, jy))
        if strength > best_strength:
            best_pixels, best_strength = component, strength

    indices = np.asarray(best_pixels, dtype=int)
    ix, iy = indices[:, 0], indices[:, 1]
    weights = smoothed[ix, iy]
    total = float(weights.sum())
    return {
        'x_mm': float(np.dot(weights, x_centers[ix]) / total),
        'y_mm': float(np.dot(weights, y_centers[iy]) / total),
        'selected_pixels': int(selected.sum()),
        'component_pixels': len(best_pixels),
        'touches_boundary': bool(np.any((ix == 0) | (ix == nx - 1) |
                                       (iy == 0) | (iy == ny - 1))),
    }


def localize_sliding_window(image: np.ndarray, x_centers: np.ndarray,
                            y_centers: np.ndarray) -> dict:
    """在固定 24×24 像素窗口中找累计散射强度最大的区域。

窗口长度约为 135 mm，覆盖当前模拟中的最大 U 形目标并留边距；
因此它使用了“目标不会超过此尺度”的先验，但不使用真值掩膜或位置。
较大窗口能把 U 的两臂和底部一起纳入，不会只选中一条高亮边。
"""
    if image.ndim != 2 or image.shape != (len(x_centers), len(y_centers)):
        raise ValueError('image shape must match x/y bin centers')
    if np.any(image < 0) or not np.isfinite(image).all():
        raise ValueError('image must contain finite nonnegative intensities')
    if WINDOW_BINS > min(image.shape) or not np.any(image > 0):
        return {'x_mm': np.nan, 'y_mm': np.nan, 'window_start_x': -1,
                'window_start_y': -1, 'window_weight_fraction': np.nan}
    windows = np.lib.stride_tricks.sliding_window_view(image, (WINDOW_BINS, WINDOW_BINS))
    masses = windows.sum(axis=(-2, -1))
    start_x, start_y = np.unravel_index(np.argmax(masses), masses.shape)
    patch = image[start_x:start_x + WINDOW_BINS, start_y:start_y + WINDOW_BINS]
    weight = float(patch.sum())
    x_grid, y_grid = np.meshgrid(x_centers[start_x:start_x + WINDOW_BINS],
                                  y_centers[start_y:start_y + WINDOW_BINS], indexing='ij')
    return {
        'x_mm': float((patch * x_grid).sum() / weight),
        'y_mm': float((patch * y_grid).sum() / weight),
        'window_start_x': int(start_x), 'window_start_y': int(start_y),
        'window_weight_fraction': weight / float(image.sum()),
    }


def localize_multiscale_window(image: np.ndarray, x_centers: np.ndarray,
                               y_centers: np.ndarray) -> dict:
    """在多个正方形窗口中，选择归一化超额散射最显著的一个。

    评分 = (窗口平均强度 - 全图平均强度) × 窗口边长（像素）。
    窗口边长 = sqrt(窗口面积)，相当于用面积平方根归一化总超额散射。
    这样大窗口不会仅因为面积大而总能赢，小窗口也不会仅因局部
    高热点而必然胜出。评分和位置完全来自图像，不使用目标真值。
    """
    if image.ndim != 2 or image.shape != (len(x_centers), len(y_centers)):
        raise ValueError('image shape must match x/y bin centers')
    if np.any(image < 0) or not np.isfinite(image).all():
        raise ValueError('image must contain finite nonnegative intensities')
    if not np.any(image > 0):
        return {'x_mm': np.nan, 'y_mm': np.nan, 'window_bins': -1,
                'window_start_x': -1, 'window_start_y': -1,
                'window_weight_fraction': np.nan, 'score': np.nan}

    # 三点平滑仅用于决定窗口；坐标仍由未平滑的原图计算。
    smoothed = smooth_3x3(image)
    background = float(smoothed.mean())
    best = None
    for size in MULTISCALE_WINDOW_BINS:
        if size > min(image.shape):
            continue
        windows = np.lib.stride_tricks.sliding_window_view(smoothed, (size, size))
        means = windows.mean(axis=(-2, -1))
        scores = (means - background) * size
        start_x, start_y = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[start_x, start_y])
        if best is None or score > best[0]:
            best = (score, size, int(start_x), int(start_y))
    if best is None:
        raise ValueError('all candidate windows exceed image dimensions')
    score, size, start_x, start_y = best
    patch = image[start_x:start_x + size, start_y:start_y + size]
    weight = float(patch.sum())
    if weight == 0:
        return {'x_mm': np.nan, 'y_mm': np.nan, 'window_bins': size,
                'window_start_x': start_x, 'window_start_y': start_y,
                'window_weight_fraction': 0.0, 'score': score}
    x_grid, y_grid = np.meshgrid(x_centers[start_x:start_x + size],
                                  y_centers[start_y:start_y + size], indexing='ij')
    return {'x_mm': float((patch * x_grid).sum() / weight),
            'y_mm': float((patch * y_grid).sum() / weight),
            'window_bins': size, 'window_start_x': start_x,
            'window_start_y': start_y,
            'window_weight_fraction': weight / float(image.sum()),
            'score': score}


def localize_robust_multiscale_window(image: np.ndarray, x_centers: np.ndarray,
                                      y_centers: np.ndarray) -> dict:
    """先限制极亮像素影响，再运行同一套多尺度窗口规则。

    强度上限取当前图像所有正像素的 95% 分位数；这是图像自身的统计量，
    不需要知道目标中心、尺寸或仿真真值。限制像素强度同时作用于窗口
    选择和窗口内质心，避免单个异常散射点决定最终位置。
    """
    if image.ndim != 2 or image.shape != (len(x_centers), len(y_centers)):
        raise ValueError('image shape must match x/y bin centers')
    if np.any(image < 0) or not np.isfinite(image).all():
        raise ValueError('image must contain finite nonnegative intensities')
    positive = image[image > 0]
    if len(positive) == 0:
        result = localize_multiscale_window(image, x_centers, y_centers)
        result['clip_value'] = np.nan
        return result
    cap = float(np.quantile(positive, ROBUST_POSITIVE_QUANTILE))
    result = localize_multiscale_window(np.minimum(image, cap), x_centers, y_centers)
    result['clip_value'] = cap
    return result
