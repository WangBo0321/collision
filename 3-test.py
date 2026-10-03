"""
Final Layer-2 test assessment
=============================
Evaluate the already-selected Layer-1 stable TEST subset using:
1) Raw Deep Ensemble (Gaussian interval)
2) Independent Conformal Prediction
3) Joint Conformal Prediction

Outputs:
- R2 / MAE / RMSE
- marginal coverage for eta_head and eta_mid_max
- joint coverage
- MPIW for eta_head, eta_mid_max, and their average
- detailed per-sample intervals for all three methods
- out-of-sample calibration-curve data
- final nominal-capacity screening based on eta_lim = 1
- confusion counts against FE truth, including false-safe FN
- publication-ready figures and CSV data

IMPORTANT:
- This script does NOT train/tune the DE.
- This script does NOT recalibrate q_lift.
- This script does NOT recompute q_ind or q_joint.
- All q values are read from the independent stable Cal-L2 set.
"""

import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# Plot style
# ============================================================
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

FONT_TICK = 14
FONT_LABEL = 16
FONT_LEGEND = 13
LINE_WIDTH = 2.0
SPINE_WIDTH = 1.5
TICK_WIDTH = 1.5
TICK_LENGTH = 6
DPI = 300

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "model_path": r"deep_ensemble_eta.pkl",

    # Already-selected Layer-1 stable TEST subset
    "test_stable_x_path": r"D-test1.csv",
    "test_stable_y_path": r"E-test1.csv",

    # Files produced by the Cal-L2 conformal calibration script
    "quantile_path": r"results_cp_cal2_stable\conformal_quantiles.csv",
    "quantile_grid_path": r"results_cp_cal2_stable\conformal_quantile_grid.csv",

    "result_dir": "results_layer2_final_test",

    "eta_limit": 1.0,
    "sigma_eps": 1e-8,
    "device": "auto",
}

OUTPUT_NAMES = ["eta_head", "eta_mid_max"]


# ============================================================
# Classes required to load the trained DE pickle
# ============================================================
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
            raise RuntimeError("The loaded Deep Ensemble contains no ensemble members.")

        X_sc = self.scaler_X.transform(X)
        X_t = torch.tensor(X_sc, dtype=torch.float32, device=self.device)

        all_means, all_vars = [], []

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

        std_total = std_total_sc * scale
        std_ep = std_ep_sc * scale
        std_al = std_al_sc * scale

        return mean, std_total, std_ep, std_al


# ============================================================
# Utilities
# ============================================================
def style_axis(ax):
    ax.tick_params(
        axis="both", which="major",
        labelsize=FONT_TICK,
        direction="in",
        top=False, right=False,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


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


def check_xy(X, Y):
    if len(X) != len(Y):
        raise ValueError(f"Stable TEST X/Y mismatch: X={len(X)}, Y={len(Y)}")
    if Y.shape[1] < 2:
        raise ValueError("Stable TEST Y must contain eta_head and eta_mid_max.")


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


def load_conformal_quantiles(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Conformal quantile file not found: {path}")

    df = pd.read_csv(path)
    required = [
        "alpha",
        "target_coverage",
        "q_ind_eta_head",
        "q_ind_eta_mid_max",
        "q_joint",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing quantile columns: " + ", ".join(missing))

    row = df.iloc[0]

    q = {
        "alpha": float(row["alpha"]),
        "target_coverage": float(row["target_coverage"]),
        "q_ind_eta_head": float(row["q_ind_eta_head"]),
        "q_ind_eta_mid_max": float(row["q_ind_eta_mid_max"]),
        "q_joint": float(row["q_joint"]),
    }

    print("\n" + "=" * 72)
    print("Fixed conformal quantiles from independent Cal-L2")
    print("=" * 72)
    print(f"Target coverage         : {q['target_coverage']:.4f}")
    print(f"q_ind_eta_head          : {q['q_ind_eta_head']:.8f}")
    print(f"q_ind_eta_mid_max       : {q['q_ind_eta_mid_max']:.8f}")
    print(f"q_joint                 : {q['q_joint']:.8f}")

    return q


# ============================================================
# Point-prediction accuracy
# ============================================================
def evaluate_point_accuracy(Y, mean, result_dir):
    rows = []

    print("\n" + "=" * 72)
    print("DE point-prediction accuracy on stable TEST subset")
    print("=" * 72)

    for j, name in enumerate(OUTPUT_NAMES):
        yt, yp = Y[:, j], mean[:, j]

        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        print(f"{name}: R2={r2:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

        rows.append({
            "output": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(result_dir, "test_point_prediction_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    return df


# ============================================================
# Build 95% intervals
# ============================================================
def build_intervals(mean, std_total, quantiles):
    target = quantiles["target_coverage"]
    alpha = 1.0 - target

    # Raw DE Gaussian interval
    z_raw = norm.ppf(1.0 - alpha / 2.0)
    raw_half = z_raw * std_total

    # Independent CP
    q_ind = np.array([
        quantiles["q_ind_eta_head"],
        quantiles["q_ind_eta_mid_max"],
    ])
    ind_half = std_total * q_ind[None, :]

    # Joint CP
    q_joint = quantiles["q_joint"]
    joint_half = q_joint * std_total

    return {
        "Raw DE": {
            "lower": mean - raw_half,
            "upper": mean + raw_half,
            "half_width": raw_half,
        },
        "Independent CP": {
            "lower": mean - ind_half,
            "upper": mean + ind_half,
            "half_width": ind_half,
        },
        "Joint CP": {
            "lower": mean - joint_half,
            "upper": mean + joint_half,
            "half_width": joint_half,
        },
    }


def interval_metrics(Y, lower, upper):
    covered = (Y >= lower) & (Y <= upper)
    marginal = covered.mean(axis=0)

    joint_covered = np.all(covered, axis=1)
    joint_cov = float(joint_covered.mean())

    widths = upper - lower
    mpiw_each = widths.mean(axis=0)
    avg_mpiw = float(widths.mean())

    return {
        "covered": covered,
        "marginal_coverage": marginal,
        "joint_covered": joint_covered,
        "joint_coverage": joint_cov,
        "widths": widths,
        "mpiw_each": mpiw_each,
        "avg_mpiw": avg_mpiw,
    }


# ============================================================
# Raw DE / Independent CP / Joint CP comparison
# ============================================================
def evaluate_interval_methods(Y, intervals, target_coverage, result_dir):
    results = {}
    rows = []

    print("\n" + "=" * 72)
    print("Interval comparison on stable TEST subset")
    print("=" * 72)

    for method, dat in intervals.items():
        m = interval_metrics(Y, dat["lower"], dat["upper"])
        results[method] = m

        print(f"\n{method}")
        print(f"  eta_head coverage    = {m['marginal_coverage'][0]:.4f}")
        print(f"  eta_mid_max coverage = {m['marginal_coverage'][1]:.4f}")
        print(f"  Joint coverage       = {m['joint_coverage']:.4f}")
        print(f"  eta_head MPIW        = {m['mpiw_each'][0]:.6f}")
        print(f"  eta_mid_max MPIW     = {m['mpiw_each'][1]:.6f}")
        print(f"  Average MPIW         = {m['avg_mpiw']:.6f}")

        rows.append({
            "Method": method,
            "eta_head coverage": m["marginal_coverage"][0],
            "eta_mid_max coverage": m["marginal_coverage"][1],
            "Joint coverage": m["joint_coverage"],
            "eta_head MPIW": m["mpiw_each"][0],
            "eta_mid_max MPIW": m["mpiw_each"][1],
            "Avg. MPIW": m["avg_mpiw"],
            "Target coverage": target_coverage,
            "n_test": len(Y),
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(result_dir, "table_interval_method_comparison.csv"),
        index=False, encoding="utf-8-sig"
    )

    return results, df


# ============================================================
# Detailed per-sample export
# ============================================================
def export_detailed_predictions(
    Y, mean, std_total, std_ep, std_al,
    intervals, interval_results, result_dir
):
    n = len(Y)

    data = {
        "stable_test_index": np.arange(1, n + 1),

        "eta_head_true": Y[:, 0],
        "eta_head_pred": mean[:, 0],
        "eta_head_abs_error": np.abs(Y[:, 0] - mean[:, 0]),
        "eta_head_std_total": std_total[:, 0],
        "eta_head_std_ep": std_ep[:, 0],
        "eta_head_std_al": std_al[:, 0],

        "eta_mid_max_true": Y[:, 1],
        "eta_mid_max_pred": mean[:, 1],
        "eta_mid_max_abs_error": np.abs(Y[:, 1] - mean[:, 1]),
        "eta_mid_max_std_total": std_total[:, 1],
        "eta_mid_max_std_ep": std_ep[:, 1],
        "eta_mid_max_std_al": std_al[:, 1],
    }

    prefixes = {
        "Raw DE": "raw",
        "Independent CP": "ind",
        "Joint CP": "joint",
    }

    for method, prefix in prefixes.items():
        lo = intervals[method]["lower"]
        hi = intervals[method]["upper"]
        hw = intervals[method]["half_width"]
        covered = interval_results[method]["covered"]
        joint_covered = interval_results[method]["joint_covered"]

        data.update({
            f"{prefix}_eta_head_lower": lo[:, 0],
            f"{prefix}_eta_head_upper": hi[:, 0],
            f"{prefix}_eta_head_half_width": hw[:, 0],
            f"{prefix}_eta_head_covered": covered[:, 0].astype(int),

            f"{prefix}_eta_mid_max_lower": lo[:, 1],
            f"{prefix}_eta_mid_max_upper": hi[:, 1],
            f"{prefix}_eta_mid_max_half_width": hw[:, 1],
            f"{prefix}_eta_mid_max_covered": covered[:, 1].astype(int),

            f"{prefix}_joint_covered": joint_covered.astype(int),
        })

    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(result_dir, "test_predictions_intervals_all_methods.csv"),
        index=False, encoding="utf-8-sig"
    )
    return df


# ============================================================
# Calibration curves
# ============================================================
def evaluate_calibration_curves(Y, mean, std_total, grid_path, result_dir):
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Quantile grid not found: {grid_path}")

    grid = pd.read_csv(grid_path)

    required = [
        "expected_coverage",
        "q_ind_eta_head",
        "q_ind_eta_mid_max",
        "q_joint",
    ]
    missing = [c for c in required if c not in grid.columns]
    if missing:
        raise ValueError("Missing quantile-grid columns: " + ", ".join(missing))

    rows = []

    for _, row in grid.iterrows():
        p = float(row["expected_coverage"])
        qh = row["q_ind_eta_head"]
        qm = row["q_ind_eta_mid_max"]
        qj = row["q_joint"]

        if pd.isna(qh) or pd.isna(qm) or pd.isna(qj):
            continue

        # Raw DE
        z = norm.ppf((1.0 + p) / 2.0)
        raw_m = interval_metrics(
            Y,
            mean - z * std_total,
            mean + z * std_total
        )

        # Independent CP
        q_ind = np.array([float(qh), float(qm)])
        ind_m = interval_metrics(
            Y,
            mean - std_total * q_ind[None, :],
            mean + std_total * q_ind[None, :]
        )

        # Joint CP
        qj = float(qj)
        joint_m = interval_metrics(
            Y,
            mean - qj * std_total,
            mean + qj * std_total
        )

        rows.append({
            "expected_coverage": p,

            "raw_eta_head_coverage": raw_m["marginal_coverage"][0],
            "raw_eta_mid_max_coverage": raw_m["marginal_coverage"][1],
            "raw_joint_coverage": raw_m["joint_coverage"],

            "ind_eta_head_coverage": ind_m["marginal_coverage"][0],
            "ind_eta_mid_max_coverage": ind_m["marginal_coverage"][1],
            "ind_joint_coverage": ind_m["joint_coverage"],

            "jointcp_eta_head_coverage": joint_m["marginal_coverage"][0],
            "jointcp_eta_mid_max_coverage": joint_m["marginal_coverage"][1],
            "jointcp_joint_coverage": joint_m["joint_coverage"],

            "q_ind_eta_head": float(qh),
            "q_ind_eta_mid_max": float(qm),
            "q_joint": qj,
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(result_dir, "data_calibration_curves_test.csv"),
        index=False, encoding="utf-8-sig"
    )
    return df


# ============================================================
# Final nominal-capacity screening
# ============================================================
def capacity_screening(Y, mean, joint_lower, joint_upper, eta_limit, result_dir):
    """
    Four-zone nominal-capacity screening based on the 95% Joint CP UPPER bounds.

    Zone I   : U_head <= 1 and U_mid <= 1
               -> both responses remain within nominal capacity after calibration

    Zone II  : U_head > 1 and U_mid <= 1
               -> head-car nominal-capacity exceedance cannot be excluded

    Zone III : U_head <= 1 and U_mid > 1
               -> intermediate-car nominal-capacity exceedance cannot be excluded

    Zone IV  : U_head > 1 and U_mid > 1
               -> exceedance cannot be excluded for both responses

    IMPORTANT:
    These are screening zones, not four physical failure modes.
    """
    true_head_exceed = Y[:, 0] > eta_limit
    true_mid_exceed = Y[:, 1] > eta_limit
    true_joint_exceed = true_head_exceed | true_mid_exceed

    # Conservative flags based on Joint CP upper bounds
    head_flag = joint_upper[:, 0] > eta_limit
    mid_flag = joint_upper[:, 1] > eta_limit
    joint_flag = head_flag | mid_flag

    # Four screening zones
    zone = np.empty(len(Y), dtype=object)

    zone[(~head_flag) & (~mid_flag)] = "Zone I: within nominal capacity"
    zone[( head_flag) & (~mid_flag)] = "Zone II: head-capacity verification"
    zone[(~head_flag) & ( mid_flag)] = "Zone III: intermediate-capacity verification"
    zone[( head_flag) & ( mid_flag)] = "Zone IV: dual-capacity verification"

    # Two-level operational decision retained for confusion / false-safe analysis
    two_level = np.where(
        joint_flag,
        "FE_verification",
        "within_nominal_capacity"
    )

    detail = pd.DataFrame({
        "stable_test_index": np.arange(1, len(Y) + 1),
        "eta_limit": eta_limit,

        "eta_head_true": Y[:, 0],
        "eta_head_pred": mean[:, 0],
        "eta_head_lower": joint_lower[:, 0],
        "eta_head_upper": joint_upper[:, 0],
        "eta_head_true_exceed": true_head_exceed.astype(int),
        "eta_head_flag_by_upper": head_flag.astype(int),

        "eta_mid_max_true": Y[:, 1],
        "eta_mid_max_pred": mean[:, 1],
        "eta_mid_max_lower": joint_lower[:, 1],
        "eta_mid_max_upper": joint_upper[:, 1],
        "eta_mid_max_true_exceed": true_mid_exceed.astype(int),
        "eta_mid_max_flag_by_upper": mid_flag.astype(int),

        "true_joint_exceed": true_joint_exceed.astype(int),
        "joint_flag_by_upper": joint_flag.astype(int),

        "capacity_zone": zone,
        "screening_decision": two_level,
    })

    detail.to_csv(
        os.path.join(result_dir, "final_capacity_screening.csv"),
        index=False, encoding="utf-8-sig"
    )

    zone_order = [
        "Zone I: within nominal capacity",
        "Zone II: head-capacity verification",
        "Zone III: intermediate-capacity verification",
        "Zone IV: dual-capacity verification",
    ]

    summary_rows = []
    for category in zone_order:
        count = int(np.sum(zone == category))
        summary_rows.append({
            "zone": category,
            "count": count,
            "ratio": float(count / len(Y)),
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(result_dir, "capacity_four_zone_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    return {
        "head_flag": head_flag,
        "mid_flag": mid_flag,
        "joint_flag": joint_flag,

        "true_head_exceed": true_head_exceed,
        "true_mid_exceed": true_mid_exceed,
        "true_joint_exceed": true_joint_exceed,

        "capacity_zone": zone,
        "two_level": two_level,
    }


# ============================================================
# Confusion / false-safe analysis
# ============================================================
def binary_confusion(true_exceed, predicted_flag):
    true_exceed = np.asarray(true_exceed, dtype=bool)
    predicted_flag = np.asarray(predicted_flag, dtype=bool)

    tp = int(np.sum(true_exceed & predicted_flag))
    fp = int(np.sum(~true_exceed & predicted_flag))
    tn = int(np.sum(~true_exceed & ~predicted_flag))
    fn = int(np.sum(true_exceed & ~predicted_flag))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    false_safe_rate_all = fn / len(true_exceed) if len(true_exceed) > 0 else np.nan
    false_safe_rate_exceed = fn / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN_false_safe": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_safe_rate_all": false_safe_rate_all,
        "false_safe_rate_among_true_exceedance": false_safe_rate_exceed,
    }


def export_confusion_analysis(screening, result_dir):
    cases = [
        ("eta_head", screening["true_head_exceed"], screening["head_flag"]),
        ("eta_mid_max", screening["true_mid_exceed"], screening["mid_flag"]),
        ("joint", screening["true_joint_exceed"], screening["joint_flag"]),
    ]

    rows = []

    print("\n" + "=" * 72)
    print("Nominal-capacity screening vs FE truth")
    print("=" * 72)

    for name, truth, flag in cases:
        m = binary_confusion(truth, flag)
        rows.append({"output": name, **m})

        print(
            f"{name}: TP={m['TP']}, FP={m['FP']}, "
            f"TN={m['TN']}, FN(false-safe)={m['FN_false_safe']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(result_dir, "capacity_screening_confusion.csv"),
        index=False, encoding="utf-8-sig"
    )
    return df


# ============================================================
# Figures
# ============================================================
def plot_coverage_comparison(summary_df, result_dir):
    methods = summary_df["Method"].tolist()
    x = np.arange(len(methods))
    w = 0.24

    fig, ax = plt.subplots(figsize=(8, 6))

    b1 = ax.bar(
        x - w, summary_df["eta_head coverage"], width=w,
        label=r"$\eta_{\mathrm{head}}$"
    )
    b2 = ax.bar(
        x, summary_df["eta_mid_max coverage"], width=w,
        label=r"$\eta_{\mathrm{mid,max}}$"
    )
    b3 = ax.bar(
        x + w, summary_df["Joint coverage"], width=w,
        label="Joint"
    )

    target = float(summary_df["Target coverage"].iloc[0])

    ax.axhline(
        target, linestyle="--", linewidth=LINE_WIDTH,
        label=f"Target = {target:.2f}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Coverage", fontsize=FONT_LABEL)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    style_axis(ax)

    for bars in [b1, b2, b3]:
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(result_dir, "fig_coverage_comparison.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()


def plot_mpiw_comparison(summary_df, result_dir):
    methods = summary_df["Method"].tolist()
    x = np.arange(len(methods))
    w = 0.24

    fig, ax = plt.subplots(figsize=(8, 6))

    b1 = ax.bar(
        x - w, summary_df["eta_head MPIW"], width=w,
        label=r"$\eta_{\mathrm{head}}$"
    )
    b2 = ax.bar(
        x, summary_df["eta_mid_max MPIW"], width=w,
        label=r"$\eta_{\mathrm{mid,max}}$"
    )
    b3 = ax.bar(
        x + w, summary_df["Avg. MPIW"], width=w,
        label="Average"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("MPIW", fontsize=FONT_LABEL)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    style_axis(ax)

    ymax = max(
        summary_df["eta_head MPIW"].max(),
        summary_df["eta_mid_max MPIW"].max(),
        summary_df["Avg. MPIW"].max(),
    )
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)

    for bars in [b1, b2, b3]:
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + ymax * 0.02,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(result_dir, "fig_mpiw_comparison.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()


def plot_one_calibration_curve(curve_df, y_columns, labels, ylabel, filename, result_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Actual nominal-coverage levels available in conformal_quantile_grid.csv
    x = curve_df["expected_coverage"].to_numpy(dtype=float)

    # Add the origin only for plotting so that the calibration curve starts at (0, 0).
    # NOTE: points below the minimum x in quantile_grid are not fabricated here.
    x_plot = np.insert(x, 0, 0.0)

    # Ideal calibration line over the full [0, 1] range
    ax.plot(
        [0.0, 1.0], [0.0, 1.0],
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Ideal"
    )

    for col, label in zip(y_columns, labels):
        y = curve_df[col].to_numpy(dtype=float)
        y_plot = np.insert(y, 0, 0.0)

        ax.plot(
            x_plot,
            y_plot,
            marker="o",
            markersize=5,
            linewidth=LINE_WIDTH,
            label=label
        )

    ax.set_xlabel("Nominal coverage", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

    # Display the same full 0-1 range as the previous calibration plots
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    style_axis(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(result_dir, filename),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()


def plot_calibration_curves(curve_df, result_dir):
    plot_one_calibration_curve(
        curve_df,
        [
            "raw_eta_head_coverage",
            "ind_eta_head_coverage",
            "jointcp_eta_head_coverage",
        ],
        ["Raw DE", "Independent CP", "Joint CP"],
        r"Empirical coverage of $\eta_{\mathrm{head}}$",
        "fig_calibration_curve_eta_head.png",
        result_dir
    )

    plot_one_calibration_curve(
        curve_df,
        [
            "raw_eta_mid_max_coverage",
            "ind_eta_mid_max_coverage",
            "jointcp_eta_mid_max_coverage",
        ],
        ["Raw DE", "Independent CP", "Joint CP"],
        r"Empirical coverage of $\eta_{\mathrm{mid,max}}$",
        "fig_calibration_curve_eta_mid_max.png",
        result_dir
    )

    plot_one_calibration_curve(
        curve_df,
        [
            "raw_joint_coverage",
            "ind_joint_coverage",
            "jointcp_joint_coverage",
        ],
        ["Raw DE", "Independent CP", "Joint CP"],
        "Empirical joint coverage",
        "fig_calibration_curve_joint.png",
        result_dir
    )


def plot_capacity_four_zones(
    Y, mean, joint_upper, capacity_zone, eta_limit, result_dir
):
    """
    Plot the four Layer-2 screening zones using the Joint CP upper bounds:
      x = U_head
      y = U_mid,max

    The lines x=1 and y=1 define the four nominal-capacity screening regions.
    """
    x = joint_upper[:, 0]
    y = joint_upper[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Threshold lines
    ax.axvline(
        eta_limit, linestyle="--", linewidth=LINE_WIDTH,
        label=r"$\eta_{\mathrm{head}}=1$"
    )
    ax.axhline(
        eta_limit, linestyle="--", linewidth=LINE_WIDTH,
        label=r"$\eta_{\mathrm{mid,max}}=1$"
    )

    zone_order = [
        "Zone I: within nominal capacity",
        "Zone II: head-capacity verification",
        "Zone III: intermediate-capacity verification",
        "Zone IV: dual-capacity verification",
    ]

    markers = ["o", "s", "^", "D"]

    for zone_name, marker in zip(zone_order, markers):
        mask = capacity_zone == zone_name
        if np.any(mask):
            ax.scatter(
                x[mask], y[mask],
                s=42,
                marker=marker,
                alpha=0.80,
                label=f"{zone_name.split(':')[0]} (n={mask.sum()})",
                zorder=3
            )

    # Set limits with enough room around the threshold
    x_min = min(float(np.min(x)), eta_limit)
    x_max = max(float(np.max(x)), eta_limit)
    y_min = min(float(np.min(y)), eta_limit)
    y_max = max(float(np.max(y)), eta_limit)

    x_pad = max(0.03, 0.08 * (x_max - x_min + 1e-12))
    y_pad = max(0.01, 0.08 * (y_max - y_min + 1e-12))

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Region labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    left_x = (xmin + eta_limit) / 2
    right_x = (eta_limit + xmax) / 2
    lower_y = (ymin + eta_limit) / 2
    upper_y = (eta_limit + ymax) / 2

    ax.text(
        left_x, lower_y, "Zone I",
        ha="center", va="center",
        fontsize=FONT_LEGEND
    )
    ax.text(
        right_x, lower_y, "Zone II",
        ha="center", va="center",
        fontsize=FONT_LEGEND
    )
    ax.text(
        left_x, upper_y, "Zone III",
        ha="center", va="center",
        fontsize=FONT_LEGEND
    )
    ax.text(
        right_x, upper_y, "Zone IV",
        ha="center", va="center",
        fontsize=FONT_LEGEND
    )

    ax.set_xlabel(
        r"Joint CP upper bound of $\eta_{\mathrm{head}}$",
        fontsize=FONT_LABEL
    )
    ax.set_ylabel(
        r"Joint CP upper bound of $\eta_{\mathrm{mid,max}}$",
        fontsize=FONT_LABEL
    )

    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    style_axis(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(result_dir, "fig_capacity_four_zones.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()

    # Export plotting data
    pd.DataFrame({
        "stable_test_index": np.arange(1, len(Y) + 1),
        "eta_head_true": Y[:, 0],
        "eta_mid_max_true": Y[:, 1],
        "eta_head_pred": mean[:, 0],
        "eta_mid_max_pred": mean[:, 1],
        "eta_head_joint_upper": joint_upper[:, 0],
        "eta_mid_max_joint_upper": joint_upper[:, 1],
        "capacity_zone": capacity_zone,
    }).to_csv(
        os.path.join(result_dir, "data_capacity_four_zones.csv"),
        index=False, encoding="utf-8-sig"
    )


def plot_joint_cp_interval(
    Y, mean, lower, upper,
    output_index, ylabel,
    filename, data_filename,
    result_dir
):
    yt = Y[:, output_index]
    yp = mean[:, output_index]
    lo = lower[:, output_index]
    hi = upper[:, output_index]

    order = np.argsort(yt)

    yt_s = yt[order]
    yp_s = yp[order]
    lo_s = lo[order]
    hi_s = hi[order]

    covered_s = (yt_s >= lo_s) & (yt_s <= hi_s)
    x = np.arange(1, len(yt_s) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(
        x, lo_s, hi_s,
        alpha=0.22,
        label="Joint CP interval"
    )

    ax.plot(
        x, yp_s,
        linewidth=LINE_WIDTH,
        label="Prediction",
        zorder=3
    )

    ax.scatter(
        x[covered_s], yt_s[covered_s],
        s=22, label="FE", zorder=4
    )

    if (~covered_s).any():
        ax.scatter(
            x[~covered_s], yt_s[~covered_s],
            marker="x", s=45,
            label="Uncovered FE",
            zorder=5
        )

    ax.set_xlabel("Selected test sample (sorted)", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    style_axis(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(result_dir, filename),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()

    pd.DataFrame({
        "sorted_position": x,
        "original_stable_test_index": order + 1,
        "true": yt_s,
        "prediction": yp_s,
        "lower": lo_s,
        "upper": hi_s,
        "covered": covered_s.astype(int),
    }).to_csv(
        os.path.join(result_dir, data_filename),
        index=False, encoding="utf-8-sig"
    )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Running script: {os.path.abspath(__file__)}")
    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    # 1. Load already-selected stable TEST subset
    print("\nLoading already-selected stable TEST data...")

    X_df = read_numeric_csv_auto(CONFIG["test_stable_x_path"])
    Y_df = read_numeric_csv_auto(CONFIG["test_stable_y_path"])

    X_test = X_df.to_numpy(dtype=np.float64)
    Y_test = Y_df.iloc[:, :2].to_numpy(dtype=np.float64)

    check_xy(X_test, Y_test)

    print(f"Stable TEST X shape: {X_test.shape}")
    print(f"Stable TEST Y shape: {Y_test.shape}")
    print("Y1 = eta_head")
    print("Y2 = eta_mid_max")

    # 2. Load trained DE
    model = load_de_model(CONFIG["model_path"], CONFIG["device"])

    # 3. Predict stable TEST + uncertainty
    print("\nPredicting stable TEST samples...")
    mean, std_total, std_ep, std_al = model.predict(X_test)

    # 4. Point-prediction accuracy
    evaluate_point_accuracy(
        Y_test, mean, CONFIG["result_dir"]
    )

    # 5. Load fixed q values from independent Cal-L2
    quantiles = load_conformal_quantiles(CONFIG["quantile_path"])

    # 6. Build three 95% interval methods
    intervals = build_intervals(mean, std_total, quantiles)

    # 7. Coverage / Joint coverage / MPIW
    interval_results, summary_df = evaluate_interval_methods(
        Y_test,
        intervals,
        quantiles["target_coverage"],
        CONFIG["result_dir"]
    )

    # 8. Detailed sample-level interval data
    export_detailed_predictions(
        Y_test, mean,
        std_total, std_ep, std_al,
        intervals, interval_results,
        CONFIG["result_dir"]
    )

    # 9. Out-of-sample calibration curves
    curve_df = evaluate_calibration_curves(
        Y_test, mean, std_total,
        CONFIG["quantile_grid_path"],
        CONFIG["result_dir"]
    )

    # 10. Final screening based on 95% Joint CP
    joint_lower = intervals["Joint CP"]["lower"]
    joint_upper = intervals["Joint CP"]["upper"]

    screening = capacity_screening(
        Y_test, mean,
        joint_lower, joint_upper,
        CONFIG["eta_limit"],
        CONFIG["result_dir"]
    )

    # 11. FE-truth comparison / false-safe analysis
    confusion_df = export_confusion_analysis(
        screening,
        CONFIG["result_dir"]
    )

    # 12. Figures
    plot_coverage_comparison(
        summary_df,
        CONFIG["result_dir"]
    )

    plot_mpiw_comparison(
        summary_df,
        CONFIG["result_dir"]
    )

    plot_capacity_four_zones(
        Y_test,
        mean,
        joint_upper,
        screening["capacity_zone"],
        CONFIG["eta_limit"],
        CONFIG["result_dir"]
    )

    plot_calibration_curves(
        curve_df,
        CONFIG["result_dir"]
    )

    plot_joint_cp_interval(
        Y_test, mean,
        joint_lower, joint_upper,
        0,
        r"$\eta_{\mathrm{head}}$",
        "fig_joint_cp_interval_eta_head.png",
        "data_joint_cp_interval_eta_head.csv",
        CONFIG["result_dir"]
    )

    plot_joint_cp_interval(
        Y_test, mean,
        joint_lower, joint_upper,
        1,
        r"$\eta_{\mathrm{mid,max}}$",
        "fig_joint_cp_interval_eta_mid_max.png",
        "data_joint_cp_interval_eta_mid_max.csv",
        CONFIG["result_dir"]
    )

    # 13. Final summary
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(f"Stable TEST samples       : {len(Y_test)}")
    print(f"Target coverage           : {quantiles['target_coverage']:.4f}")
    print(f"q_ind_eta_head            : {quantiles['q_ind_eta_head']:.6f}")
    print(f"q_ind_eta_mid_max         : {quantiles['q_ind_eta_mid_max']:.6f}")
    print(f"q_joint                   : {quantiles['q_joint']:.6f}")

    joint_row = summary_df[
        summary_df["Method"] == "Joint CP"
    ].iloc[0]

    print(f"Joint CP eta_head coverage: {joint_row['eta_head coverage']:.4f}")
    print(f"Joint CP eta_mid coverage : {joint_row['eta_mid_max coverage']:.4f}")
    print(f"Joint CP joint coverage   : {joint_row['Joint coverage']:.4f}")
    print(f"Joint CP head MPIW        : {joint_row['eta_head MPIW']:.6f}")
    print(f"Joint CP mid MPIW         : {joint_row['eta_mid_max MPIW']:.6f}")
    print(f"Joint CP average MPIW     : {joint_row['Avg. MPIW']:.6f}")

    zone_names = [
        "Zone I: within nominal capacity",
        "Zone II: head-capacity verification",
        "Zone III: intermediate-capacity verification",
        "Zone IV: dual-capacity verification",
    ]

    print("\nFour-zone capacity screening:")
    for zone_name in zone_names:
        count = int(np.sum(screening["capacity_zone"] == zone_name))
        print(f"  {zone_name:<43}: {count}")

    joint_conf = confusion_df[
        confusion_df["output"] == "joint"
    ].iloc[0]

    print(f"Joint false-safe count    : {int(joint_conf['FN_false_safe'])}")
    print(f"Results directory         : {os.path.abspath(CONFIG['result_dir'])}")

    print("\nGenerated key files:")
    for name in [
        "test_point_prediction_metrics.csv",
        "table_interval_method_comparison.csv",
        "test_predictions_intervals_all_methods.csv",
        "data_calibration_curves_test.csv",
        "final_capacity_screening.csv",
        "capacity_four_zone_summary.csv",
        "data_capacity_four_zones.csv",
        "capacity_screening_confusion.csv",
        "fig_coverage_comparison.png",
        "fig_mpiw_comparison.png",
        "fig_capacity_four_zones.png",
        "fig_calibration_curve_eta_head.png",
        "fig_calibration_curve_eta_mid_max.png",
        "fig_calibration_curve_joint.png",
        "fig_joint_cp_interval_eta_head.png",
        "fig_joint_cp_interval_eta_mid_max.png",
    ]:
        print("  " + name)