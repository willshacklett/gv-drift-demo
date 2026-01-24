import math
import random
import json
from pathlib import Path

import matplotlib.pyplot as plt

from gv import compute_gv, load_baseline, predict, mse, r2_score


def make_dataset(n, d, seed=7):
    random.seed(seed)
    true_coef = [1.0, -2.0, 0.5, 3.0][:d]
    true_bias = 0.25

    xs, ys = [], []
    for _ in range(n):
        row = [random.uniform(-2, 2) for _ in range(d)]
        noise = random.gauss(0, 0.25)
        y = sum(w * x for w, x in zip(true_coef, row)) + true_bias + noise
        xs.append(row)
        ys.append(y)
    return xs, ys


def apply_drift(coef, bias, step):
    drift = 0.0025 * step
    wobble = 0.02 * math.sin(step / 50)
    new_coef = [
        c + (drift if i % 2 == 0 else -drift) + wobble
        for i, c in enumerate(coef)
    ]
    return new_coef, bias + 0.25 * drift


baseline = load_baseline("baseline.json")
xs, ys = make_dataset(600, len(baseline.coef))

steps = range(101)
gvs, r2s = [], []

for step in steps:
    coef, bias = apply_drift(baseline.coef, baseline.bias, step)
    yhat = predict(xs, coef, bias)
    gvs.append(compute_gv(coef, bias, baseline))
    r2s.append(r2_score(ys, yhat))

gv_alert = 0.08
test_threshold = 0.92

first_gv = next((i for i, g in enumerate(gvs) if g > gv_alert), None)
first_fail = next((i for i, r in enumerate(r2s) if r < test_threshold), None)

plt.plot(steps, r2s, label="R²")
plt.plot(steps, gvs, label="GV")
plt.axhline(test_threshold, linestyle="--", label="Test threshold")
plt.axhline(gv_alert, linestyle="--", label="GV alert")

if first_gv is not None:
    plt.axvline(first_gv, linestyle=":")
if first_fail is not None:
    plt.axvline(first_fail, linestyle=":")

plt.legend()
plt.title("GV flags drift before tests fail")
plt.savefig("output.png", dpi=160)
plt.close()

Path("results.json").write_text(json.dumps({
    "first_gv_alert": first_gv,
    "first_test_failure": first_fail
}, indent=2))
