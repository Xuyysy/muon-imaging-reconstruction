"""将已有事件行收敛为只含探测器观测的不可变记录。

本模块是数据边界，不是稳定的科研算法 API。它有意丢弃 PoCA、角度、真值、
材料标签等派生/评价字段，防止新算法推断过程意外依赖旧重建结果或答案标签。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


HIT_PLANES = ("up1", "up2", "down1", "down2")
HIT_AXES = ("x", "y", "z")
HIT_COLUMNS = tuple(
    f"{plane}_{axis}_mm" for plane in HIT_PLANES for axis in HIT_AXES
)


@dataclass(frozen=True)
class ObservedEvent:
    """单条四层击中观测；坐标单位固定为 mm。"""

    event_id: int
    hits_mm: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.hits_mm) != len(HIT_PLANES):
            raise ValueError("exactly four detector-plane hits are required")
        if any(len(hit) != len(HIT_AXES) for hit in self.hits_mm):
            raise ValueError("each detector hit must contain x, y, and z")
        if not all(math.isfinite(value) for hit in self.hits_mm for value in hit):
            raise ValueError("hit coordinates must be finite")


def observed_event_from_mapping(row: Mapping[str, object]) -> ObservedEvent:
    """只从 CSV-like mapping 读取事件编号与四层击中坐标。

    额外列（包括 poca_*、theta_*、truth/material 等）会被忽略；输入 mapping
    不会被修改。事件 ID 不要求全局唯一，因为不同 Geant4 run 可从 0 重新编号。
    """

    required = ("event",) + HIT_COLUMNS
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"missing observed input columns: {', '.join(missing)}")

    try:
        event_id = int(row["event"])
        values = tuple(float(row[name]) for name in HIT_COLUMNS)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("event ID and hit coordinates must be numeric") from exc

    hits = tuple(
        tuple(values[offset : offset + len(HIT_AXES)])
        for offset in range(0, len(values), len(HIT_AXES))
    )
    return ObservedEvent(event_id=event_id, hits_mm=hits)


def observed_events_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[ObservedEvent, ...]:
    """把输入行转换为不可变观测记录，不保留输入容器或派生列引用。"""

    return tuple(observed_event_from_mapping(row) for row in rows)
