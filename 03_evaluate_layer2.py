#!/usr/bin/env python3
"""Evaluate Raw DE, Independent CP and Joint CP on the held-out stable test subset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from deep_ensemble import DeepEnsemble, load_xy  # noqa: E402


def interval_metrics(Y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    covered = (Y >= lower) & (Y <= upper)
    width = upper - lower
    return {
        "marginal": covered.mean(axis=0),
        "joint": float(np.all(covered, axis=1).mean()),
        "mpiw": width.mean(axis=0),
        "covered": covered,
    }


def label_category(head: np.ndarray, mid: np.ndarray) -> np.ndarray:
    out = np.empty(len(head), dtype=object)
    out[(~head) & (~mid)] = "Unflagged"
    out[head & (~mid)] = "Head-car flagged"
    out[(~head) & mid] = "Mid-car flagged"
    out[head & mid] = "Jointly flagged"
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/deep_ensemble_eta.pt")
    p.add_argument("--test-x", required=True)
    p.add_argument("--test-y", required=True)
    p.add_argument("--quantiles", default="results/layer2_calibration/conformal_quantiles.csv")
    p.add_argument("--eta-limit", type=float, default=1.0)
    p.add_argument("--result-dir", default="results/layer2_test")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X, Y = load_xy(args.test_x, args.test_y)
    model = DeepEnsemble.load(args.model, device=args.device)
    mean, std_total, std_ep, std_al = model.predict(X)

    qdf = pd.read_csv(args.quantiles)
    q = qdf.iloc[0]
    target = float(q["target_coverage"])
    alpha = 1 - target
    z = norm.ppf(1 - alpha / 2)

    intervals = {
        "Raw DE": (mean - z * std_total, mean + z * std_total),
        "Independent CP": (
            mean - std_total * np.array([q["q_ind_eta_head"], q["q_ind_eta_mid_max"]]),
            mean + std_total * np.array([q["q_ind_eta_head"], q["q_ind_eta_mid_max"]]),
        ),
        "Joint CP": (
            mean - float(q["q_joint"]) * std_total,
            mean + float(q["q_joint"]) * std_total,
        ),
    }

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    point_rows = []
    for j, name in enumerate(["eta_head", "eta_mid_max"]):
        point_rows.append(
            {
                "output": name,
                "R2": r2_score(Y[:, j], mean[:, j]),
                "MAE": mean_absolute_error(Y[:, j], mean[:, j]),
                "RMSE": float(np.sqrt(np.mean((Y[:, j] - mean[:, j]) ** 2))),
            }
        )
    pd.DataFrame(point_rows).to_csv(out / "point_prediction_metrics.csv", index=False)

    metric_rows = []
    metrics_by_method = {}
    for method, (lower, upper) in intervals.items():
        m = interval_metrics(Y, lower, upper)
        metrics_by_method[method] = m
        metric_rows.append(
            {
                "Method": method,
                "eta_head coverage": m["marginal"][0],
                "eta_mid_max coverage": m["marginal"][1],
                "Joint coverage": m["joint"],
                "eta_head MPIW": m["mpiw"][0],
                "eta_mid_max MPIW": m["mpiw"][1],
                "Target coverage": target,
                "n_test": len(Y),
            }
        )
    pd.DataFrame(metric_rows).to_csv(out / "interval_method_comparison.csv", index=False)

    joint_lower, joint_upper = intervals["Joint CP"]
    true_head = Y[:, 0] > args.eta_limit
    true_mid = Y[:, 1] > args.eta_limit
    flag_head = joint_upper[:, 0] > args.eta_limit
    flag_mid = joint_upper[:, 1] > args.eta_limit
    true_category = label_category(true_head, true_mid)
    pred_category = label_category(flag_head, flag_mid)

    detail = pd.DataFrame(
        {
            "sample_index": np.arange(1, len(Y) + 1),
            "eta_head_true": Y[:, 0],
            "eta_head_pred": mean[:, 0],
            "eta_head_std_total": std_total[:, 0],
            "eta_head_joint_lower": joint_lower[:, 0],
            "eta_head_joint_upper": joint_upper[:, 0],
            "eta_mid_max_true": Y[:, 1],
            "eta_mid_max_pred": mean[:, 1],
            "eta_mid_max_std_total": std_total[:, 1],
            "eta_mid_max_joint_lower": joint_lower[:, 1],
            "eta_mid_max_joint_upper": joint_upper[:, 1],
            "FE_reference_category": true_category,
            "screening_category": pred_category,
        }
    )
    detail.to_csv(out / "joint_cp_screening_details.csv", index=False)

    cats = ["Unflagged", "Head-car flagged", "Mid-car flagged", "Jointly flagged"]
    confusion = pd.crosstab(
        pd.Categorical(true_category, categories=cats),
        pd.Categorical(pred_category, categories=cats),
        dropna=False,
    )
    confusion.index.name = "FE reference category"
    confusion.columns.name = "Screening category"
    confusion.to_csv(out / "screening_confusion_table.csv")

    missed = int(np.sum((true_head | true_mid) & ~(flag_head | flag_mid)))
    print(f"Stable test samples: {len(Y)}")
    print(f"Joint CP joint coverage: {metrics_by_method['Joint CP']['joint']:.4f}")
    print(f"Observed missed capacity exceedances: {missed}")


if __name__ == "__main__":
    main()
