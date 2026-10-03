#!/usr/bin/env python3
"""Permutation-SHAP analysis of eta_head and eta_mid_max Deep-Ensemble means."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from deep_ensemble import DeepEnsemble, read_numeric_csv  # noqa: E402

# Column order used by the released 16-parameter data matrices.
FEATURE_NAMES = [
    "F_ac", "F_ea", "F_hc", "S_ac", "S_ea", "S_hc",
    "F_s1", "F_s2", "F_s3", "F_s4",
    "S_s1", "S_s2", "S_s3", "S_s4",
    "V_off", "H_off",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/deep_ensemble_eta.pt")
    p.add_argument("--background-x", required=True, help="Training-data X used only to sample SHAP background points.")
    p.add_argument("--explain-x", required=True, help="Samples to explain; for the paper use the calibrated-stable Layer-2 test subset.")
    p.add_argument("--background-size", type=int, default=100)
    p.add_argument("--n-permutations", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-dir", default="results/shap_capacity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X_bg_all = read_numeric_csv(args.background_x).to_numpy(dtype=float)
    X_exp = read_numeric_csv(args.explain_x).to_numpy(dtype=float)
    if X_bg_all.shape[1] != 16 or X_exp.shape[1] != 16:
        raise ValueError("SHAP inputs must have 16 columns in the documented feature order.")

    rng = np.random.default_rng(args.seed)
    n_bg = min(args.background_size, len(X_bg_all))
    X_bg = X_bg_all[rng.choice(len(X_bg_all), size=n_bg, replace=False)]
    model = DeepEnsemble.load(args.model, device="cpu")

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)

    for output_index, output_name in enumerate(["eta_head", "eta_mid_max"]):
        def predict_fn(X):
            mean, _, _, _ = model.predict(np.asarray(X, dtype=float))
            return mean[:, output_index]

        masker = shap.maskers.Independent(X_bg)
        explainer = shap.Explainer(
            predict_fn, masker, algorithm="permutation", feature_names=FEATURE_NAMES
        )
        max_evals = (2 * X_exp.shape[1] + 1) * args.n_permutations
        sv = explainer(X_exp, max_evals=max_evals)

        pd.DataFrame(sv.values, columns=FEATURE_NAMES).to_csv(
            out / f"shap_values_{output_name}.csv", index=False
        )
        importance = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "mean_abs_SHAP": np.abs(sv.values).mean(axis=0),
                "mean_SHAP": sv.values.mean(axis=0),
            }
        ).sort_values("mean_abs_SHAP", ascending=False)
        importance.to_csv(out / f"shap_importance_{output_name}.csv", index=False)

        # Export the exact feature values used in the SHAP analysis.
        values = pd.DataFrame(X_exp, columns=[f"{x}_value" for x in FEATURE_NAMES])
        shap_df = pd.DataFrame(sv.values, columns=[f"{x}_SHAP" for x in FEATURE_NAMES])
        pd.concat([values, shap_df], axis=1).to_csv(
            out / f"shap_beeswarm_data_{output_name}.csv", index=False
        )

    print(f"SHAP results: {out.resolve()}")


if __name__ == "__main__":
    main()
