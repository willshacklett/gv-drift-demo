from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Baseline:
    coef: List[float]
    bias: float


def load_baseline(path: str | Path) -> Baseline:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    coef = data.get("coef")
    bias = data.get("bias")
    if not isinstance(coef, list):
        raise ValueError("baseline.json must contain list field 'coef'")
    if not isinstance(bias, (int, float)):
        raise ValueError("baseline.json must contain numeric field 'bias'")
    return Baseline(coef=[float(x) for x in coef], bias=float(bias))


def _l2(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def compute_gv(current_coef: List[float], current_bias: float, baseline: Baseline) -> float:
    diff = [c - b for c, b in zip(current_coef, baseline.coef)]
    denom = _l2(baseline.coef) + 1e-9
    coef_drift = _l2(diff) / denom

    bias_drift = abs(current_bias - baseline.bias) / (abs(baseline.bias) + 1e-9)
    return float(coef_drift + 0.25 * bias_drift)


def predict(xs: List[List[float]], coef: List[float], bias: float) -> List[float]:
    return [sum(w * x for w, x in zip(coef, row)) + bias for row in xs]


def mse(y_true: List[float], y_pred: List[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((a - mean_y) ** 2 for a in y_true) + 1e-12
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return float(1.0 - (ss_res / ss_tot))
