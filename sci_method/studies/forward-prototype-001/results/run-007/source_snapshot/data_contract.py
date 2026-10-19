"""将已有事件行收敛为只含探测器观测的不可变记录。

本模块是数据边界，不是稳定的科研算法 API。它有意丢弃 PoCA、角度、真值、
材料标签等派生/评价字段，防止新算法推断过程意外依赖旧重建结果或答案标签。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Mapping, Sequence


HIT_PLANES = ("up1", "up2", "down1", "down2")
HIT_AXES = ("x", "y", "z")
HIT_COLUMNS = tuple(
    f"{plane}_{axis}_mm" for plane in HIT_PLANES for axis in HIT_AXES
)


@dataclass(frozen=True)
class ObservedEvent:
    """单条四层击中观测；坐标单位固定为 mm。"""

    run_id: str
    event_id: int
    hits_mm: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if isinstance(self.event_id, bool) or not isinstance(self.event_id, int):
            raise ValueError("event ID must be an integer")
        if len(self.hits_mm) != len(HIT_PLANES):
            raise ValueError("exactly four detector-plane hits are required")
        if any(len(hit) != len(HIT_AXES) for hit in self.hits_mm):
            raise ValueError("each detector hit must contain x, y, and z")
        if not all(math.isfinite(value) for hit in self.hits_mm for value in hit):
            raise ValueError("hit coordinates must be finite")

    @property
    def key(self) -> tuple[str, int]:
        """Cross-run identity; event IDs are only unique within one run."""
        return self.run_id, self.event_id


def _strict_integer_event_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("event ID must be an integer")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value and (value.isdecimal() or (value[0] in "+-" and value[1:].isdecimal())):
            return int(value)
    raise ValueError("event ID must be an integer")


def observed_event_from_mapping(row: Mapping[str, object], *, run_id: str) -> ObservedEvent:
    """只从 CSV-like mapping 读取事件编号与四层击中坐标。

    额外列（包括 poca_*、theta_*、truth/material 等）会被忽略；输入 mapping
    不会被修改。事件 ID 严格要求整数；调用方必须传入 run_id，跨运行用
    (run_id, event_id) 联合标识。
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    required = ("event",) + HIT_COLUMNS
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"missing observed input columns: {', '.join(missing)}")

    event_id = _strict_integer_event_id(row["event"])
    try:
        values = tuple(float(row[name]) for name in HIT_COLUMNS)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hit coordinates must be numeric") from exc

    hits = tuple(
        tuple(values[offset : offset + len(HIT_AXES)])
        for offset in range(0, len(values), len(HIT_AXES))
    )
    return ObservedEvent(run_id=run_id.strip(), event_id=event_id, hits_mm=hits)


def observed_events_from_rows(
    rows: Sequence[Mapping[str, object]], *, run_id: str,
) -> tuple[ObservedEvent, ...]:
    """转换为不可变记录；同一批输入不允许重复的联合事件键。"""

    events = tuple(observed_event_from_mapping(row, run_id=run_id) for row in rows)
    keys = tuple(event.key for event in events)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate event key in input batch")
    return events
