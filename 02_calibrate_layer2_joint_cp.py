#!/usr/bin/env python3
"""Calibrate Independent CP and Joint CP on the independent stable Cal-L2 subset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from deep_ensemble import DeepEnsemble, load_xy  # noqa: E402


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration score array is empty.")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        raise ValueError(f"n={n} is too small for alpha={alpha}; k={k} > n.")
    return float(np.sort(scores)[k - 1]), k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/deep_ensemble_eta.pt")
    p.add_argument("--cal-x", required=True)
    p.add_argument("--cal-y", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--sigma-eps", type=float, default=1e-8)
    p.add_argument("--result-dir", default="results/layer2_calibration")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X, Y = load_xy(args.cal_x, args.cal_y)
    model = DeepEnsemble.load(args.model, device=args.device)
    mean, std_total, std_ep, std_al = model.predict(X)

    sigma = np.maximum(std_total, args.sigma_eps)
    scores = np.abs(Y - mean) / sigma
    s_head = scores[:, 0]
    s_mid = scores[:, 1]
    s_joint = np.maximum(s_head, s_mid)

    q_head, k_head = finite_sample_quantile(s_head, args.alpha)
    q_mid, k_mid = finite_sample_quantile(s_mid, args.alpha)
    q_joint, k_joint = finite_sample_quantile(s_joint, args.alpha)

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "calibration_index": np.arange(1, len(Y) + 1),
            "eta_head_true": Y[:, 0],
            "eta_head_pred": mean[:, 0],
            "eta_head_std_total": std_total[:, 0],
            "eta_head_std_ep": std_ep[:, 0],
            "eta_head_std_al": std_al[:, 0],
            "eta_head_score": s_head,
            "eta_mid_max_true": Y[:, 1],
            "eta_mid_max_pred": mean[:, 1],
            "eta_mid_max_std_total": std_total[:, 1],
            "eta_mid_max_std_ep": std_ep[:, 1],
            "eta_mid_max_std_al": std_al[:, 1],
            "eta_mid_max_score": s_mid,
            "joint_score": s_joint,
        }
    ).to_csv(out / "calibration_scores.csv", index=False)

    pd.DataFrame(
        [
            {
                "alpha": args.alpha,
                "target_coverage": 1 - args.alpha,
                "n_calibration": len(Y),
                "k_ind_eta_head": k_head,
                "q_ind_eta_head": q_head,
                "k_ind_eta_mid_max": k_mid,
                "q_ind_eta_mid_max": q_mid,
                "k_joint": k_joint,
                "q_joint": q_joint,
            }
        ]
    ).to_csv(out / "conformal_quantiles.csv", index=False)

    # Grid for out-of-sample calibration-curve evaluation.
    rows = []
    for coverage in np.arange(0.05, 0.951, 0.05):
        alpha = 1 - coverage
        row = {"nominal_coverage": coverage, "alpha": alpha}
        try:
            row["q_ind_eta_head"], _ = finite_sample_quantile(s_head, alpha)
            row["q_ind_eta_mid_max"], _ = finite_sample_quantile(s_mid, alpha)
            row["q_joint"], _ = finite_sample_quantile(s_joint, alpha)
        except ValueError:
            row["q_ind_eta_head"] = np.nan
            row["q_ind_eta_mid_max"] = np.nan
            row["q_joint"] = np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "conformal_quantile_grid.csv", index=False)

    print(f"n_cal = {len(Y)}")
    print(f"q_ind_eta_head = {q_head:.6f}")
    print(f"q_ind_eta_mid_max = {q_mid:.6f}")
    print(f"q_joint = {q_joint:.6f}")


if __name__ == "__main__":
    main()
