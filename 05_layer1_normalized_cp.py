#!/usr/bin/env python3
"""Layer-1 amplitude-normalized conformal calibration and three-region screening.

Input CSV files must contain columns:
    pred_max, true_max
where each row represents one design sample and each value is the global maximum
wheelset lift extracted from the 16 monitored wheelset-lift channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        raise ValueError(f"Insufficient calibration samples: n={n}, k={k}.")
    return float(np.sort(scores)[k - 1]), k


def exact_one_sided_upper(x: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0:
        return np.nan
    if x == 0:
        return 1 - (1 - confidence) ** (1 / n)
    if x >= n:
        return 1.0
    return float(beta.ppf(confidence, x + 1, n - x))


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pred_max", "true_max"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    out = df.copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="raise")
    return out


def classify(pred: np.ndarray, margin: np.ndarray, limit: float) -> np.ndarray:
    upper = pred + margin
    lower = pred - margin
    labels = np.full(len(pred), "to-be-verified", dtype=object)
    labels[upper <= limit] = "calibrated stable"
    labels[lower > limit] = "calibrated unstable"
    return labels


def summarize_method(
    name: str,
    pred: np.ndarray,
    true: np.ndarray,
    margin: np.ndarray,
    limit: float,
    local_lo: float,
    local_hi: float,
) -> dict:
    lower = pred - margin
    upper = pred + margin
    labels = classify(pred, margin, limit)
    covered = (true >= lower) & (true <= upper)
    local = (true >= local_lo) & (true <= local_hi)
    true_unstable = true > limit
    true_stable = ~true_unstable
    false_stable = true_unstable & (labels == "calibrated stable")
    false_unstable = true_stable & (labels == "calibrated unstable")
    return {
        "Method": name,
        "Overall coverage": covered.mean(),
        "Local coverage": covered[local].mean() if local.any() else np.nan,
        "Local n": int(local.sum()),
        "Stable": int(np.sum(labels == "calibrated stable")),
        "To-be-verified": int(np.sum(labels == "to-be-verified")),
        "Unstable": int(np.sum(labels == "calibrated unstable")),
        "False stable count": int(false_stable.sum()),
        "True unstable n": int(true_unstable.sum()),
        "False stable upper 95%": exact_one_sided_upper(int(false_stable.sum()), int(true_unstable.sum())),
        "False unstable count": int(false_unstable.sum()),
        "True stable n": int(true_stable.sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--limit", type=float, default=20.0)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--local-lo", type=float, default=15.0)
    p.add_argument("--local-hi", type=float, default=25.0)
    p.add_argument("--result-dir", default="results/layer1_normalized_cp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cal = load_table(args.calib)
    test = load_table(args.test)

    c_pred = cal["pred_max"].to_numpy(float)
    c_true = cal["true_max"].to_numpy(float)
    abs_scores = np.abs(c_true - c_pred)
    norm_scores = abs_scores / np.maximum(c_pred, args.delta)

    q_norm, k_norm = finite_sample_quantile(norm_scores, args.alpha)
    q_abs, k_abs = finite_sample_quantile(abs_scores, args.alpha)

    pred = test["pred_max"].to_numpy(float)
    true = test["true_max"].to_numpy(float)
    norm_margin = q_norm * np.maximum(pred, args.delta)
    abs_margin = np.full_like(pred, q_abs)

    normalized = summarize_method(
        "Normalized CP", pred, true, norm_margin, args.limit, args.local_lo, args.local_hi
    )
    global_cp = summarize_method(
        "Global CP", pred, true, abs_margin, args.limit, args.local_lo, args.local_hi
    )
    normalized["Margin at limit"] = q_norm * args.limit
    global_cp["Margin at limit"] = q_abs

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([global_cp, normalized]).to_csv(out / "cp_method_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "alpha": args.alpha,
                "n_calibration": len(cal),
                "k_norm": k_norm,
                "q_norm": q_norm,
                "k_global": k_abs,
                "q_global_mm": q_abs,
                "delta_mm": args.delta,
                "wheelset_lift_limit_mm": args.limit,
            }
        ]
    ).to_csv(out / "conformal_quantiles.csv", index=False)

    lower = pred - norm_margin
    upper = pred + norm_margin
    labels = classify(pred, norm_margin, args.limit)
    covered = (true >= lower) & (true <= upper)
    detail = test.copy()
    detail["normalized_margin"] = norm_margin
    detail["interval_lower"] = lower
    detail["interval_upper"] = upper
    detail["covered"] = covered.astype(int)
    detail["screening_category"] = labels
    detail.to_csv(out / "normalized_cp_test_results.csv", index=False)

    pd.DataFrame(
        {
            "calibration_index": np.arange(1, len(cal) + 1),
            "pred_max": c_pred,
            "true_max": c_true,
            "absolute_error_mm": abs_scores,
            "normalized_score": norm_scores,
        }
    ).to_csv(out / "normalized_cp_calibration_scores.csv", index=False)

    # Figure 1: absolute error versus FE response amplitude.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(c_true, abs_scores, s=24, alpha=0.75)
    ax.axvline(args.limit, linestyle="--", linewidth=1.5)
    ax.axvspan(args.local_lo, args.local_hi, alpha=0.08)
    ax.set_xlabel("FE maximum wheelset lift (mm)")
    ax.set_ylabel("Absolute maximum-lift error (mm)")
    fig.tight_layout()
    fig.savefig(out / "error_vs_amplitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: normalized calibration-score distribution.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(norm_scores, bins="auto", density=True, alpha=0.75)
    ax.axvline(q_norm, linestyle="--", linewidth=1.5, label=rf"$q_{{norm}}={q_norm:.3f}$")
    ax.set_xlabel("Amplitude-normalized nonconformity score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "normalized_score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: calibrated screening intervals sorted by prediction.
    order = np.argsort(pred)
    x = np.arange(1, len(pred) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, lower[order], upper[order], alpha=0.15, label="Calibrated interval")
    ax.plot(x, pred[order], linewidth=1.5, label="Prediction")
    ax.axhline(args.limit, linestyle="--", linewidth=1.5, label=f"Limit = {args.limit:g} mm")
    ax.set_xlabel("Test samples sorted by predicted maximum lift")
    ax.set_ylabel("Maximum wheelset lift (mm)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "normalized_cp_screening.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"q_norm = {q_norm:.8f}")
    print(f"q_global = {q_abs:.8f} mm")
    print(f"Normalized CP coverage = {normalized['Overall coverage']:.4f}")
    print(
        "False-stable one-sided 95% upper bound among truly unstable samples = "
        f"{normalized['False stable upper 95%']:.4f}"
    )


if __name__ == "__main__":
    main()
