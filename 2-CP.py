"""
Layer-2 conformal calibration on the already-selected stable Cal-L2 set
=======================================================================
This script:
1) loads the trained Deep Ensemble;
2) predicts eta_head and eta_mid_max plus DE uncertainty;
3) computes independent conformal quantiles for each output;
4) computes the joint conformal quantile;
5) exports score/quantile data needed by the final TEST script.

It DOES NOT evaluate coverage or MPIW, because those must be evaluated
on the independent TEST set, not on the calibration set.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CONFIG = {
    "model_path": r"deep_ensemble_eta.pkl",

    # Already-selected stable Layer-2 calibration subset
    "cal2_stable_x_path": r"E:\Python_Document\RESS\补充数据\extracted_data.csv",
    "cal2_stable_y_path": r"E:\Python_Document\RESS\补充数据\label_0_data.csv",

    "result_dir": "results_cp_cal2_stable",
    "alpha": 0.05,
    "sigma_eps": 1e-8,
    "device": "auto",

    # Used only to export q values at multiple confidence levels.
    # The actual empirical calibration curve is calculated later on TEST.
    "curve_confidence_levels": [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95,
    ],
}

OUTPUT_NAMES = ["eta_head", "eta_mid_max"]


class MLPMember(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 256, 256), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, out_dim)
        self.log_var_head = nn.Linear(prev, out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        mean = self.mean_head(feat)
        log_var = torch.clamp(self.log_var_head(feat), -10.0, 10.0)
        return mean, log_var


class DeepEnsembleETA:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.members = []
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.out_dim = None
        self.device = "cpu"

    def predict(self, X):
        if len(self.members) == 0:
            raise RuntimeError("The loaded Deep Ensemble contains no members.")

        X_sc = self.scaler_X.transform(X)
        X_t = torch.tensor(X_sc, dtype=torch.float32, device=self.device)

        all_means = []
        all_vars = []

        with torch.no_grad():
            for model in self.members:
                model = model.to(self.device)
                model.eval()
                mu_sc, log_var_sc = model(X_t)
                all_means.append(mu_sc.detach().cpu().numpy())
                all_vars.append(np.exp(log_var_sc.detach().cpu().numpy()))

        all_means = np.asarray(all_means)
        all_vars = np.asarray(all_vars)

        mean_sc = all_means.mean(axis=0)
        var_ep_sc = all_means.var(axis=0)
        var_al_sc = all_vars.mean(axis=0)
        var_total_sc = var_ep_sc + var_al_sc

        std_ep_sc = np.sqrt(np.maximum(var_ep_sc, 0.0))
        std_al_sc = np.sqrt(np.maximum(var_al_sc, 0.0))
        std_total_sc = np.sqrt(np.maximum(var_total_sc, 0.0))

        mean = self.scaler_Y.inverse_transform(mean_sc)
        scale = self.scaler_Y.scale_

        return (
            mean,
            std_total_sc * scale,
            std_ep_sc * scale,
            std_al_sc * scale,
        )


def read_numeric_csv_auto(path):
    raw = pd.read_csv(path, header=None)
    num = raw.apply(pd.to_numeric, errors="coerce")

    if len(num) >= 2 and num.iloc[0].isna().any() and num.iloc[1:].notna().all().all():
        num = num.iloc[1:].reset_index(drop=True)

    if num.isna().any().any():
        r, c = np.argwhere(num.isna().to_numpy())[0]
        raise ValueError(
            f"{path} contains a non-numeric or missing value at row {r + 1}, col {c + 1}."
        )

    return num.astype(np.float64)


def load_de_model(path, requested_device="auto"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Deep Ensemble model not found: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    model.device = device
    for member in model.members:
        member.to(device)
        member.eval()

    print(f"Loaded Deep Ensemble: {os.path.abspath(path)}")
    print(f"Ensemble members    : {len(model.members)}")
    print(f"Prediction device   : {device}")
    return model


def finite_sample_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if n == 0:
        raise ValueError("Calibration scores are empty.")

    k = int(np.ceil((n + 1) * (1 - alpha)))

    if k > n:
        raise ValueError(
            f"n={n} is too small for alpha={alpha}: k={k} > n={n}."
        )

    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1]), k, sorted_scores


def run_calibration(Y, mean, std_total, std_ep, std_al, cfg):
    alpha = cfg["alpha"]
    eps = cfg["sigma_eps"]
    result_dir = cfg["result_dir"]
    n = len(Y)

    sigma_safe = np.maximum(std_total, eps)
    scores = np.abs(Y - mean) / sigma_safe

    score_head = scores[:, 0]
    score_mid = scores[:, 1]
    score_joint = np.maximum(score_head, score_mid)

    # Independent CP
    q_head, k_head, sorted_head = finite_sample_quantile(score_head, alpha)
    q_mid, k_mid, sorted_mid = finite_sample_quantile(score_mid, alpha)

    # Joint CP
    q_joint, k_joint, sorted_joint = finite_sample_quantile(score_joint, alpha)

    print("\n" + "=" * 72)
    print("Layer-2 conformal calibration")
    print("=" * 72)
    print(f"Stable Cal-L2 samples   : {n}")
    print(f"Target coverage         : {1-alpha:.4f}")
    print(f"Independent eta_head    : k={k_head}, q={q_head:.8f}")
    print(f"Independent eta_mid_max : k={k_mid}, q={q_mid:.8f}")
    print(f"Joint CP                : k={k_joint}, q_joint={q_joint:.8f}")

    # Prediction and score data
    pd.DataFrame({
        "calibration_index": np.arange(1, n + 1),

        "eta_head_true": Y[:, 0],
        "eta_head_pred": mean[:, 0],
        "eta_head_std_total": std_total[:, 0],
        "eta_head_std_ep": std_ep[:, 0],
        "eta_head_std_al": std_al[:, 0],
        "eta_head_abs_error": np.abs(Y[:, 0] - mean[:, 0]),
        "eta_head_score": score_head,

        "eta_mid_max_true": Y[:, 1],
        "eta_mid_max_pred": mean[:, 1],
        "eta_mid_max_std_total": std_total[:, 1],
        "eta_mid_max_std_ep": std_ep[:, 1],
        "eta_mid_max_std_al": std_al[:, 1],
        "eta_mid_max_abs_error": np.abs(Y[:, 1] - mean[:, 1]),
        "eta_mid_max_score": score_mid,

        "joint_score": score_joint,
    }).to_csv(
        os.path.join(result_dir, "cal2_stable_predictions_and_scores.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # Main quantile file: next TEST script reads this
    pd.DataFrame([{
        "alpha": alpha,
        "target_coverage": 1-alpha,
        "n_calibration": n,

        "k_ind_eta_head": k_head,
        "q_ind_eta_head": q_head,

        "k_ind_eta_mid_max": k_mid,
        "q_ind_eta_mid_max": q_mid,

        "k_joint": k_joint,
        "q_joint": q_joint,
    }]).to_csv(
        os.path.join(result_dir, "conformal_quantiles.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # Sorted scores for checking the score tails
    rank = np.arange(1, n + 1)
    pd.DataFrame({
        "rank": rank,
        "eta_head_score_sorted": sorted_head,
        "eta_mid_max_score_sorted": sorted_mid,
        "joint_score_sorted": sorted_joint,
        "is_q_ind_eta_head": (rank == k_head).astype(int),
        "is_q_ind_eta_mid_max": (rank == k_mid).astype(int),
        "is_q_joint": (rank == k_joint).astype(int),
    }).to_csv(
        os.path.join(result_dir, "conformal_scores_sorted.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # q-grid needed later to calculate calibration curves on TEST
    curve_rows = []
    for expected_cov in cfg["curve_confidence_levels"]:
        alpha_i = 1.0 - expected_cov

        try:
            qh, kh, _ = finite_sample_quantile(score_head, alpha_i)
            qm, km, _ = finite_sample_quantile(score_mid, alpha_i)
            qj, kj, _ = finite_sample_quantile(score_joint, alpha_i)

            curve_rows.append({
                "expected_coverage": expected_cov,
                "alpha": alpha_i,
                "q_ind_eta_head": qh,
                "q_ind_eta_mid_max": qm,
                "q_joint": qj,
                "k_ind_eta_head": kh,
                "k_ind_eta_mid_max": km,
                "k_joint": kj,
            })
        except ValueError:
            curve_rows.append({
                "expected_coverage": expected_cov,
                "alpha": alpha_i,
                "q_ind_eta_head": np.nan,
                "q_ind_eta_mid_max": np.nan,
                "q_joint": np.nan,
                "k_ind_eta_head": np.nan,
                "k_ind_eta_mid_max": np.nan,
                "k_joint": np.nan,
            })

    pd.DataFrame(curve_rows).to_csv(
        os.path.join(result_dir, "conformal_quantile_grid.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:")
    print("  cal2_stable_predictions_and_scores.csv")
    print("  conformal_quantiles.csv")
    print("  conformal_scores_sorted.csv")
    print("  conformal_quantile_grid.csv")

    return q_head, q_mid, q_joint


if __name__ == "__main__":
    print(f"Running script: {os.path.abspath(__file__)}")
    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    print("\nLoading already-selected stable Cal-L2 data...")

    X_df = read_numeric_csv_auto(CONFIG["cal2_stable_x_path"])
    Y_df = read_numeric_csv_auto(CONFIG["cal2_stable_y_path"])

    X = X_df.to_numpy(dtype=np.float64)
    Y = Y_df.iloc[:, :2].to_numpy(dtype=np.float64)

    if len(X) != len(Y):
        raise ValueError(f"X/Y mismatch: X={len(X)}, Y={len(Y)}")

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    model = load_de_model(CONFIG["model_path"], CONFIG["device"])

    print("\nPredicting stable Cal-L2 samples...")
    mean, std_total, std_ep, std_al = model.predict(X)

    q_head, q_mid, q_joint = run_calibration(
        Y, mean, std_total, std_ep, std_al, CONFIG
    )

    print("\n" + "=" * 72)
    print("All done")
    print("=" * 72)
    print(f"q_ind_eta_head    = {q_head:.8f}")
    print(f"q_ind_eta_mid_max = {q_mid:.8f}")
    print(f"q_joint           = {q_joint:.8f}")
    print(f"Results directory = {os.path.abspath(CONFIG['result_dir'])}")
    print()
    print("NOTE:")
    print("Coverage / Joint coverage / MPIW / calibration-curve empirical coverage")
    print("must be calculated on the independent TEST set in the next script.")