#!/usr/bin/env python3
"""Train the Layer-2 Deep Ensemble for eta_head and eta_mid_max."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from deep_ensemble import DEConfig, DeepEnsemble, load_xy  # noqa: E402


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(["eta_head", "eta_mid_max"]):
        rows.append(
            {
                "output": name,
                "R2": r2_score(y_true[:, j], y_pred[:, j]),
                "MAE": mean_absolute_error(y_true[:, j], y_pred[:, j]),
                "RMSE": float(np.sqrt(np.mean((y_true[:, j] - y_pred[:, j]) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-x", required=True)
    p.add_argument("--train-y", required=True)
    p.add_argument("--val-x")
    p.add_argument("--val-y")
    p.add_argument("--model-out", default="models/deep_ensemble_eta.pt")
    p.add_argument("--result-dir", default="results/deep_ensemble")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--dropout-min", type=float, default=0.05)
    p.add_argument("--dropout-max", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.val_x is None) != (args.val_y is None):
        raise ValueError("--val-x and --val-y must be provided together.")

    X_train, Y_train = load_xy(args.train_x, args.train_y)
    if args.val_x:
        X_val, Y_val = load_xy(args.val_x, args.val_y)
    else:
        X_val = Y_val = None

    cfg = DEConfig(
        n_models=args.n_models,
        hidden_dims=tuple(args.hidden_dims),
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        patience=args.patience,
        random_seed=args.seed,
        device=args.device,
    )

    model = DeepEnsemble(cfg)
    model.fit(X_train, Y_train, X_val, Y_val)
    model.save(args.model_out)

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist = pd.concat(model.training_histories, ignore_index=True)
    hist.to_csv(out / "training_history.csv", index=False)

    pred_train, std_train, std_ep, std_al = model.predict(X_train)
    metrics(Y_train, pred_train).to_csv(out / "train_metrics.csv", index=False)
    pd.DataFrame(
        np.column_stack([Y_train, pred_train, std_train, std_ep, std_al]),
        columns=[
            "eta_head_true", "eta_mid_max_true",
            "eta_head_pred", "eta_mid_max_pred",
            "eta_head_std_total", "eta_mid_max_std_total",
            "eta_head_std_ep", "eta_mid_max_std_ep",
            "eta_head_std_al", "eta_mid_max_std_al",
        ],
    ).to_csv(out / "train_predictions.csv", index=False)

    if X_val is not None:
        pred_val, std_val, _, _ = model.predict(X_val)
        metrics(Y_val, pred_val).to_csv(out / "validation_metrics.csv", index=False)
        pd.DataFrame(
            np.column_stack([Y_val, pred_val, std_val]),
            columns=[
                "eta_head_true", "eta_mid_max_true",
                "eta_head_pred", "eta_mid_max_pred",
                "eta_head_std_total", "eta_mid_max_std_total",
            ],
        ).to_csv(out / "validation_predictions.csv", index=False)

    print(f"Saved model: {Path(args.model_out).resolve()}")
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
