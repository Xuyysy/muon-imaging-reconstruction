"""Ablation and finite-precision diagnostics for the frozen run-001 record."""

from __future__ import annotations

import math
import sys


def _truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def decision_flags_from_row(row: dict) -> dict[str, bool]:
    """Parse decision flags consistently from CSV strings or Python booleans."""
    return {
        name: _truth(row[f"decision_{name}"])
        for name in (
            "lrt_only", "bic_only", "estimability_plus_lrt",
            "estimability_power_only", "estimability_lrt_power", "full_composite",
        )
    }


def ablation_decisions(row: dict) -> dict[str, bool]:
    """Apply prespecified nested gates to an existing trial row, no refitting."""
    converged = _truth(row["coarse_converged"]) and _truth(row["fine_converged"])
    lrt = float(row["lrt_train"])
    calibrated_cutoff = float(row["lrt_threshold"])
    estimable = _truth(row["contrast_estimable"])
    power_ok = float(row["predicted_power"]) >= 0.80
    lrt_only = converged and lrt > calibrated_cutoff
    estimability_lrt = lrt_only and estimable
    estimation_power = estimable and power_ok
    estimability_lrt_power = estimability_lrt and power_ok
    n_target = sum(int(row[key]) for key in ("n_left", "n_right", "n_other"))
    n_total_likelihood = n_target
    if row["background_mode"] == "empty_field_calibrated" and row.get("empty_field_seed"):
        n_total_likelihood += n_target
    bic_only = converged and lrt > math.log(max(n_total_likelihood, 2))
    return {
        "lrt_only": lrt_only,
        "bic_only": bic_only,
        "estimability_plus_lrt": estimability_lrt,
        "estimability_power_only": estimation_power,
        "estimability_lrt_power": estimability_lrt_power,
        "full_composite": converged and estimability_lrt_power,
    }


def lrt_numerical_diagnostic(lrt: float, objective_scale: float) -> dict:
    """Describe a near-zero LRT relative to objective scale; never alter a cutoff."""
    lrt, scale = abs(float(lrt)), abs(float(objective_scale))
    if not math.isfinite(lrt) or not math.isfinite(scale) or scale == 0.0:
        raise ValueError("LRT and nonzero objective scale must be finite")
    relative = lrt / scale
    return {
        "absolute_lrt": lrt,
        "objective_scale_proxy": scale,
        "relative_to_objective_scale": relative,
        "machine_epsilon_units": relative / sys.float_info.epsilon,
        "threshold_changed": False,
        "interpretation": "Scale diagnostic only. For exactly equivalent nested models, this near-zero difference is optimizer/floating-point residue, not resolved statistical evidence.",
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, center-half), min(1.0, center+half)

