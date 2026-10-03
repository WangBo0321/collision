import os
import pickle
import random
import warnings
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# Plot style
# ============================================================
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

FONT_TICK = 14
FONT_LABEL = 16
FONT_LEGEND = 12
LINE_WIDTH = 2.0
SPINE_WIDTH = 1.5
TICK_WIDTH = 1.5
TICK_LENGTH = 6
DPI = 300

RAW_INTERVAL_COVERAGE = 0.95


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


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Original 1000 samples
    "x_path": r"D.csv",
    "y_path": r"E:\Python_Document\RESS\1new_mod\eta_mid_variance_adjust/E_eta_mid_max_modified.csv",

    # Output
    "result_dir": "results_de_eta",
    "model_path": "deep_ensemble_eta.pkl",

    # Explicit sequential allocation of the original 1000 samples
    # Python slicing:
    #   [0:800]   -> rows 1-800
    #   [800:900] -> rows 801-900, reserved for Layer-1 calibration
    #   [900:1000]-> rows 901-1000, independent test
    "train_start": 0,
    "train_end": 1041,
    "test_start": 1041,
    "test_end": 1141,

    # Optional additional Layer-2 calibration pool.
    # After the new FE samples are generated, set these two paths.
    # Leave as None for now and the script will simply skip this part.
    "cal2_x_path": None,
    "cal2_y_path": None,

    # Deep Ensemble
    "n_models": 5,
    "hidden_dims": (256, 256, 256),
    "dropout": 0.10,
    "lr": 1e-4,
    "epochs": 300,
    "batch_size": 64,
    "weight_decay": 1e-3,
    "random_seed": 42,
    "device": "auto",
}

OUTPUT_NAMES = ["eta_head", "eta_mid_max"]


# ============================================================
# Utilities
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_numeric_csv_auto(path):
    """
    Read a numeric CSV with or without a header.
    """
    raw = pd.read_csv(path, header=None)
    num = raw.apply(pd.to_numeric, errors="coerce")

    # Header row detection
    if (
        len(num) >= 2
        and num.iloc[0].isna().any()
        and num.iloc[1:].notna().all().all()
    ):
        num = num.iloc[1:].reset_index(drop=True)

    if num.isna().any().any():
        r, c = np.argwhere(num.isna().to_numpy())[0]
        raise ValueError(
            f"{path} contains a non-numeric or missing value "
            f"at row {r + 1}, col {c + 1}."
        )

    return num.astype(np.float64)


def check_xy(X, Y, name="dataset"):
    if len(X) != len(Y):
        raise ValueError(
            f"{name} X/Y sample count mismatch: X={len(X)}, Y={len(Y)}"
        )

    if Y.shape[1] < 2:
        raise ValueError(
            f"{name} Y must contain at least two columns: "
            "eta_head and eta_mid_max."
        )


# ============================================================
# Deep Ensemble member
# ============================================================
class MLPMember(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims=(256, 256, 256),
        dropout=0.1
    ):
        super().__init__()

        layers = []
        prev = in_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        self.backbone = nn.Sequential(*layers)

        # Predictive mean
        self.mean_head = nn.Linear(prev, out_dim)

        # Heteroscedastic log-variance
        self.log_var_head = nn.Linear(prev, out_dim)

    def forward(self, x):
        feat = self.backbone(x)

        mean = self.mean_head(feat)

        log_var = torch.clamp(
            self.log_var_head(feat),
            -10.0,
            10.0
        )

        return mean, log_var


# ============================================================
# Gaussian NLL
# ============================================================
def gaussian_nll_loss(mean, log_var, target, lambda_var=8):
    var = torch.exp(log_var)

    nll = 0.5 * (
        log_var
        + (target - mean) ** 2 / var
    )

    var_penalty = lambda_var * var.mean()

    return nll.mean() + var_penalty


# ============================================================
# Train one ensemble member
# ============================================================
def train_member(
    X_train,
    Y_train,
    in_dim,
    out_dim,
    cfg,
    device,
    seed
):
    set_seed(seed)

    model = MLPMember(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
    )

    X_t = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    Y_t = torch.tensor(
        Y_train,
        dtype=torch.float32
    )

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        TensorDataset(X_t, Y_t),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )

    history = []

    for epoch in range(cfg["epochs"]):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            mu, log_var = model(xb)

            loss = gaussian_nll_loss(
                mu,
                log_var,
                yb
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            running_loss += loss.item() * len(xb)
            n_seen += len(xb)

        scheduler.step()

        epoch_loss = running_loss / max(n_seen, 1)
        history.append({
            "epoch": epoch + 1,
            "train_NLL": epoch_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(
                f"    epoch {epoch + 1:4d}/{cfg['epochs']}  "
                f"train_NLL={epoch_loss:.6f}"
            )

    model.eval()

    return model, pd.DataFrame(history)


# ============================================================
# Pure Deep Ensemble
# ============================================================
class DeepEnsembleETA:
    def __init__(self, cfg):
        self.cfg = cfg

        self.members = []
        self.training_histories = []

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        self.out_dim = None

        if cfg["device"] == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = cfg["device"]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    def fit(self, X_train, Y_train):
        print("=" * 72)
        print("Pure Deep Ensemble training")
        print("=" * 72)

        print(f"Training samples : {len(X_train)}")
        print(f"Input dimension  : {X_train.shape[1]}")
        print(f"Output dimension : {Y_train.shape[1]}")
        print(f"Ensemble members : {self.cfg['n_models']}")
        print(f"Device           : {self.device}")

        # Fit scalers ONLY on training data
        X_train_sc = self.scaler_X.fit_transform(X_train)
        Y_train_sc = self.scaler_Y.fit_transform(Y_train)

        self.out_dim = Y_train_sc.shape[1]

        print(
            f"\nTraining {self.cfg['n_models']} ensemble members..."
        )

        for m in range(self.cfg["n_models"]):
            seed = (
                self.cfg["random_seed"]
                + 1000 * m
            )

            print(
                f"\n  member {m + 1}/{self.cfg['n_models']} "
                f" seed={seed}"
            )

            model, history = train_member(
                X_train=X_train_sc,
                Y_train=Y_train_sc,
                in_dim=X_train_sc.shape[1],
                out_dim=self.out_dim,
                cfg=self.cfg,
                device=self.device,
                seed=seed,
            )

            history.insert(0, "member", m + 1)
            history.insert(1, "seed", seed)

            self.members.append(model)
            self.training_histories.append(history)

        print("\nDeep Ensemble training completed.")

    # --------------------------------------------------------
    # Raw DE prediction
    # --------------------------------------------------------
    def predict(self, X):
        """
        Returns
        -------
        mean : [N, 2]
            Ensemble predictive mean in original Y units.

        std_total : [N, 2]
            Total predictive standard deviation.

        std_ep : [N, 2]
            Epistemic standard deviation from member disagreement.

        std_al : [N, 2]
            Aleatoric standard deviation from member-predicted variance.
        """
        if len(self.members) == 0:
            raise RuntimeError(
                "The Deep Ensemble has not been trained."
            )

        X_sc = self.scaler_X.transform(X)

        X_t = torch.tensor(
            X_sc,
            dtype=torch.float32,
            device=self.device
        )

        all_means = []
        all_vars = []

        with torch.no_grad():
            for model in self.members:
                model = model.to(self.device)
                model.eval()

                mu_sc, log_var_sc = model(X_t)

                all_means.append(
                    mu_sc.cpu().numpy()
                )

                all_vars.append(
                    np.exp(
                        log_var_sc.cpu().numpy()
                    )
                )

        # Shape:
        # [M, N, output_dim]
        all_means = np.asarray(all_means)
        all_vars = np.asarray(all_vars)

        # Ensemble mean
        mean_sc = all_means.mean(axis=0)

        # Epistemic variance:
        # variance of member means
        var_ep_sc = all_means.var(axis=0)

        # Aleatoric variance:
        # mean predicted variance
        var_al_sc = all_vars.mean(axis=0)

        # Total predictive variance
        var_total_sc = (
            var_ep_sc
            + var_al_sc
        )

        std_ep_sc = np.sqrt(
            np.maximum(var_ep_sc, 0.0)
        )

        std_al_sc = np.sqrt(
            np.maximum(var_al_sc, 0.0)
        )

        std_total_sc = np.sqrt(
            np.maximum(var_total_sc, 0.0)
        )

        # Back-transform mean
        mean = self.scaler_Y.inverse_transform(
            mean_sc
        )

        # Standard deviation rescales only by scale_
        y_scale = self.scaler_Y.scale_

        std_ep = (
            std_ep_sc
            * y_scale
        )

        std_al = (
            std_al_sc
            * y_scale
        )

        std_total = (
            std_total_sc
            * y_scale
        )

        return (
            mean,
            std_total,
            std_ep,
            std_al,
        )

    # --------------------------------------------------------
    # Member-level prediction export
    # --------------------------------------------------------
    def predict_members(self, X):
        """
        Return member-level predictive means and aleatoric standard deviations
        in original output units.

        Returns
        -------
        member_means : [M, N, output_dim]
        member_std_al : [M, N, output_dim]
        """
        if len(self.members) == 0:
            raise RuntimeError("The Deep Ensemble has not been trained.")

        X_sc = self.scaler_X.transform(X)
        X_t = torch.tensor(
            X_sc,
            dtype=torch.float32,
            device=self.device
        )

        all_means = []
        all_std_al = []

        with torch.no_grad():
            for model in self.members:
                model = model.to(self.device)
                model.eval()

                mu_sc, log_var_sc = model(X_t)

                mu_sc_np = mu_sc.detach().cpu().numpy()
                std_al_sc_np = np.exp(0.5 * log_var_sc.detach().cpu().numpy())

                mu = self.scaler_Y.inverse_transform(mu_sc_np)
                std_al = std_al_sc_np * self.scaler_Y.scale_

                all_means.append(mu)
                all_std_al.append(std_al)

        return np.asarray(all_means), np.asarray(all_std_al)


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    def save(self, path=None):
        path = (
            path
            or self.cfg["model_path"]
        )

        # Move members to CPU before pickle
        for model in self.members:
            model.cpu()

        old_device = self.device
        self.device = "cpu"

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"Model saved: {os.path.abspath(path)}")

        # Keep current object usable
        if old_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            for model in self.members:
                model.cuda()


# ============================================================
# Metrics
# ============================================================
def evaluate_predictions(
    y_true,
    y_pred,
    result_dir,
    prefix="test",
    std_total=None,
    std_ep=None,
    std_al=None,
    raw_coverage=RAW_INTERVAL_COVERAGE
):
    rows = []

    print("\n" + "=" * 72)
    print(f"{prefix} prediction accuracy")
    print("=" * 72)

    for j, name in enumerate(OUTPUT_NAMES):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        r2 = r2_score(
            yt,
            yp
        )

        mae = mean_absolute_error(
            yt,
            yp
        )

        rmse = np.sqrt(
            np.mean(
                (yt - yp) ** 2
            )
        )

        print(
            f"{name}: "
            f"R2={r2:.6f}, "
            f"MAE={mae:.6f}, "
            f"RMSE={rmse:.6f}"
        )

        row = {
            "output": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
        }

        if std_total is not None:
            sigma = std_total[:, j]
            z = norm.ppf((1.0 + raw_coverage) / 2.0)
            lower = yp - z * sigma
            upper = yp + z * sigma

            row.update({
                "mean_std_total": float(np.mean(sigma)),
                "RMV": float(np.sqrt(np.mean(sigma ** 2))),
                "RMV_RMSE_ratio": float(
                    np.sqrt(np.mean(sigma ** 2)) / (rmse + 1e-12)
                ),
                "raw_interval_coverage": float(
                    np.mean((yt >= lower) & (yt <= upper))
                ),
                "raw_MPIW": float(np.mean(upper - lower)),
            })

            if std_ep is not None:
                row["mean_std_ep"] = float(np.mean(std_ep[:, j]))
            if std_al is not None:
                row["mean_std_al"] = float(np.mean(std_al[:, j]))

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(
            result_dir,
            f"{prefix}_metrics.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    return df


# ============================================================
# Export prediction CSV
# ============================================================
def export_predictions(
    y_true,
    mean,
    std_total,
    std_ep,
    std_al,
    sample_index,
    output_path,
    raw_coverage=RAW_INTERVAL_COVERAGE
):
    """
    Export complete aggregate DE prediction information, including errors and
    the raw Gaussian prediction interval.
    """
    z = norm.ppf((1.0 + raw_coverage) / 2.0)

    lower = mean - z * std_total
    upper = mean + z * std_total
    covered = (y_true >= lower) & (y_true <= upper)

    df = pd.DataFrame({
        "sample_index": sample_index,

        "eta_head_true": y_true[:, 0],
        "eta_head_pred": mean[:, 0],
        "eta_head_error": mean[:, 0] - y_true[:, 0],
        "eta_head_abs_error": np.abs(mean[:, 0] - y_true[:, 0]),
        "eta_head_sq_error": (mean[:, 0] - y_true[:, 0]) ** 2,
        "eta_head_std_total": std_total[:, 0],
        "eta_head_std_ep": std_ep[:, 0],
        "eta_head_std_al": std_al[:, 0],
        "eta_head_raw_lower": lower[:, 0],
        "eta_head_raw_upper": upper[:, 0],
        "eta_head_raw_half_width": z * std_total[:, 0],
        "eta_head_raw_width": 2.0 * z * std_total[:, 0],
        "eta_head_raw_covered": covered[:, 0].astype(int),

        "eta_mid_max_true": y_true[:, 1],
        "eta_mid_max_pred": mean[:, 1],
        "eta_mid_max_error": mean[:, 1] - y_true[:, 1],
        "eta_mid_max_abs_error": np.abs(mean[:, 1] - y_true[:, 1]),
        "eta_mid_max_sq_error": (mean[:, 1] - y_true[:, 1]) ** 2,
        "eta_mid_max_std_total": std_total[:, 1],
        "eta_mid_max_std_ep": std_ep[:, 1],
        "eta_mid_max_std_al": std_al[:, 1],
        "eta_mid_max_raw_lower": lower[:, 1],
        "eta_mid_max_raw_upper": upper[:, 1],
        "eta_mid_max_raw_half_width": z * std_total[:, 1],
        "eta_mid_max_raw_width": 2.0 * z * std_total[:, 1],
        "eta_mid_max_raw_covered": covered[:, 1].astype(int),

        "raw_joint_covered": np.all(covered, axis=1).astype(int),
    })

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Predictions saved: "
        f"{os.path.abspath(output_path)}"
    )

    return df


# ============================================================
# Save train/test split data
# ============================================================
def export_dataset_split(
    X_train, Y_train, X_test, Y_test,
    train_index, test_index, result_dir
):
    os.makedirs(os.path.join(result_dir, "data_split"), exist_ok=True)

    x_cols = [f"X{i+1}" for i in range(X_train.shape[1])]

    train_df = pd.DataFrame(X_train, columns=x_cols)
    train_df.insert(0, "sample_index", train_index)
    train_df["eta_head"] = Y_train[:, 0]
    train_df["eta_mid_max"] = Y_train[:, 1]
    train_df.to_csv(
        os.path.join(result_dir, "data_split", "train_data_used.csv"),
        index=False, encoding="utf-8-sig"
    )

    test_df = pd.DataFrame(X_test, columns=x_cols)
    test_df.insert(0, "sample_index", test_index)
    test_df["eta_head"] = Y_test[:, 0]
    test_df["eta_mid_max"] = Y_test[:, 1]
    test_df.to_csv(
        os.path.join(result_dir, "data_split", "test_data_used.csv"),
        index=False, encoding="utf-8-sig"
    )


# ============================================================
# Save training histories
# ============================================================
def export_training_history(model, result_dir):
    if not model.training_histories:
        return None

    history_df = pd.concat(
        model.training_histories,
        ignore_index=True
    )

    history_df.to_csv(
        os.path.join(result_dir, "training_history_all_members.csv"),
        index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for member_id, sub in history_df.groupby("member", sort=True):
        ax.plot(
            sub["epoch"],
            sub["train_NLL"],
            linewidth=1.5,
            label=f"Member {member_id}"
        )

    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Training NLL", fontsize=FONT_LABEL)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_dir, "fig_training_nll_all_members.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()

    return history_df


# ============================================================
# Save member-level predictions
# ============================================================
def export_member_predictions(
    model, X, y_true, sample_index, result_dir, prefix="test"
):
    member_means, member_std_al = model.predict_members(X)

    rows = {
        "sample_index": sample_index,
        "eta_head_true": y_true[:, 0],
        "eta_mid_max_true": y_true[:, 1],
    }

    for m in range(member_means.shape[0]):
        rows[f"member{m+1}_eta_head_pred"] = member_means[m, :, 0]
        rows[f"member{m+1}_eta_head_std_al"] = member_std_al[m, :, 0]
        rows[f"member{m+1}_eta_mid_max_pred"] = member_means[m, :, 1]
        rows[f"member{m+1}_eta_mid_max_std_al"] = member_std_al[m, :, 1]

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(result_dir, f"{prefix}_member_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )
    return df


# ============================================================
# Figure 1: parity scatter with raw-DE error bars
# ============================================================
def plot_errorbar_parity(
    y_true, mean, std_total, sample_index, result_dir,
    prefix="test", raw_coverage=RAW_INTERVAL_COVERAGE
):
    z = norm.ppf((1.0 + raw_coverage) / 2.0)

    plot_dir = os.path.join(result_dir, "fig_data_errorbar_parity")
    os.makedirs(plot_dir, exist_ok=True)

    for j, name in enumerate(OUTPUT_NAMES):
        yt = y_true[:, j]
        yp = mean[:, j]
        yerr = z * std_total[:, j]

        lim_min = min(np.min(yt), np.min(yp - yerr))
        lim_max = max(np.max(yt), np.max(yp + yerr))
        pad = 0.05 * max(lim_max - lim_min, 1e-12)
        lo = lim_min - pad
        hi = lim_max + pad

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.errorbar(
            yt,
            yp,
            yerr=yerr,
            fmt="o",
            markersize=4,
            alpha=0.65,
            elinewidth=0.8,
            capsize=2,
            label=f"Raw DE {int(raw_coverage*100)}% interval",
            zorder=3
        )

        ax.plot(
            [lo, hi], [lo, hi],
            linestyle="--",
            linewidth=LINE_WIDTH,
            label="Ideal"
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("FE", fontsize=FONT_LABEL)
        ax.set_ylabel("Prediction", fontsize=FONT_LABEL)
        ax.legend(frameon=False, fontsize=FONT_LEGEND)
        style_axis(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"fig_errorbar_parity_{name}.png"),
            dpi=DPI, bbox_inches="tight"
        )
        plt.close()

        pd.DataFrame({
            "sample_index": sample_index,
            "true": yt,
            "prediction": yp,
            "std_total": std_total[:, j],
            "errorbar_half_width": yerr,
            "lower": yp - yerr,
            "upper": yp + yerr,
        }).to_csv(
            os.path.join(plot_dir, f"{name}.csv"),
            index=False, encoding="utf-8-sig"
        )


# ============================================================
# Figure 2: sorted raw-DE prediction bands
# ============================================================
def plot_sorted_prediction_band(
    y_true, mean, std_total, sample_index, result_dir,
    raw_coverage=RAW_INTERVAL_COVERAGE
):
    z = norm.ppf((1.0 + raw_coverage) / 2.0)

    plot_dir = os.path.join(result_dir, "fig_data_sorted_prediction_band")
    os.makedirs(plot_dir, exist_ok=True)

    for j, name in enumerate(OUTPUT_NAMES):
        yt = y_true[:, j]
        yp = mean[:, j]
        lower = yp - z * std_total[:, j]
        upper = yp + z * std_total[:, j]

        order = np.argsort(yt)
        x = np.arange(1, len(yt) + 1)

        yt_s = yt[order]
        yp_s = yp[order]
        lo_s = lower[order]
        hi_s = upper[order]
        idx_s = np.asarray(sample_index)[order]
        covered_s = (yt_s >= lo_s) & (yt_s <= hi_s)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.fill_between(
            x, lo_s, hi_s,
            alpha=0.22,
            label=f"Raw DE {int(raw_coverage*100)}% interval"
        )
        ax.plot(
            x, yp_s,
            linewidth=LINE_WIDTH,
            label="Prediction",
            zorder=3
        )
        ax.scatter(
            x[covered_s], yt_s[covered_s],
            s=22,
            label="FE",
            zorder=4
        )
        if (~covered_s).any():
            ax.scatter(
                x[~covered_s], yt_s[~covered_s],
                marker="x",
                s=45,
                label="Uncovered FE",
                zorder=5
            )

        ax.set_xlabel("Test sample (sorted)", fontsize=FONT_LABEL)
        ax.set_ylabel(
            r"$\eta_{\mathrm{head}}$" if j == 0
            else r"$\eta_{\mathrm{mid,max}}$",
            fontsize=FONT_LABEL
        )
        ax.legend(frameon=False, fontsize=FONT_LEGEND)
        style_axis(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"fig_sorted_prediction_band_{name}.png"),
            dpi=DPI, bbox_inches="tight"
        )
        plt.close()

        pd.DataFrame({
            "sorted_position": x,
            "sample_index": idx_s,
            "true": yt_s,
            "prediction": yp_s,
            "lower": lo_s,
            "upper": hi_s,
            "covered": covered_s.astype(int),
        }).to_csv(
            os.path.join(plot_dir, f"{name}.csv"),
            index=False, encoding="utf-8-sig"
        )


# ============================================================
# Figure 3: uncertainty vs absolute error
# ============================================================
def plot_uncertainty_vs_error(
    y_true, mean, std_total, sample_index, result_dir
):
    plot_dir = os.path.join(result_dir, "fig_data_uncertainty_vs_error")
    os.makedirs(plot_dir, exist_ok=True)

    for j, name in enumerate(OUTPUT_NAMES):
        err = np.abs(y_true[:, j] - mean[:, j])
        sigma = std_total[:, j]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            sigma, err,
            s=30,
            alpha=0.70,
            zorder=3
        )

        if len(sigma) >= 2 and np.std(sigma) > 0:
            coef = np.polyfit(sigma, err, 1)
            xfit = np.linspace(sigma.min(), sigma.max(), 100)
            ax.plot(
                xfit,
                np.polyval(coef, xfit),
                linewidth=LINE_WIDTH
            )

        ax.set_xlabel(r"$\sigma_{\mathrm{total}}$", fontsize=FONT_LABEL)
        ax.set_ylabel("Absolute error", fontsize=FONT_LABEL)
        style_axis(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"fig_uncertainty_vs_error_{name}.png"),
            dpi=DPI, bbox_inches="tight"
        )
        plt.close()

        pd.DataFrame({
            "sample_index": sample_index,
            "std_total": sigma,
            "abs_error": err,
        }).to_csv(
            os.path.join(plot_dir, f"{name}.csv"),
            index=False, encoding="utf-8-sig"
        )


# ============================================================
# Figure 4: uncertainty decomposition
# ============================================================
def plot_uncertainty_decomposition(
    std_total, std_ep, std_al, sample_index, result_dir
):
    plot_dir = os.path.join(result_dir, "fig_data_uncertainty_decomposition")
    os.makedirs(plot_dir, exist_ok=True)

    for j, name in enumerate(OUTPUT_NAMES):
        x = np.arange(1, len(std_total) + 1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, std_total[:, j], linewidth=LINE_WIDTH, label="Total")
        ax.plot(x, std_ep[:, j], linewidth=1.5, label="Epistemic")
        ax.plot(x, std_al[:, j], linewidth=1.5, label="Aleatoric")

        ax.set_xlabel("Test sample", fontsize=FONT_LABEL)
        ax.set_ylabel(r"$\sigma$", fontsize=FONT_LABEL)
        ax.legend(frameon=False, fontsize=FONT_LEGEND)
        style_axis(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"fig_uncertainty_decomposition_{name}.png"),
            dpi=DPI, bbox_inches="tight"
        )
        plt.close()

        pd.DataFrame({
            "sample_index": sample_index,
            "std_total": std_total[:, j],
            "std_ep": std_ep[:, j],
            "std_al": std_al[:, j],
        }).to_csv(
            os.path.join(plot_dir, f"{name}.csv"),
            index=False, encoding="utf-8-sig"
        )


# ============================================================
# Figure 5: residual vs prediction
# ============================================================
def plot_residuals(
    y_true, mean, sample_index, result_dir
):
    plot_dir = os.path.join(result_dir, "fig_data_residuals")
    os.makedirs(plot_dir, exist_ok=True)

    for j, name in enumerate(OUTPUT_NAMES):
        residual = mean[:, j] - y_true[:, j]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            mean[:, j],
            residual,
            s=30,
            alpha=0.70,
            zorder=3
        )
        ax.axhline(0.0, linestyle="--", linewidth=LINE_WIDTH)

        ax.set_xlabel("Prediction", fontsize=FONT_LABEL)
        ax.set_ylabel("Residual (prediction - FE)", fontsize=FONT_LABEL)
        style_axis(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"fig_residual_{name}.png"),
            dpi=DPI, bbox_inches="tight"
        )
        plt.close()

        pd.DataFrame({
            "sample_index": sample_index,
            "prediction": mean[:, j],
            "true": y_true[:, j],
            "residual": residual,
        }).to_csv(
            os.path.join(plot_dir, f"{name}.csv"),
            index=False, encoding="utf-8-sig"
        )


# ============================================================
# Save run configuration
# ============================================================
def export_run_config(cfg, result_dir):
    safe_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, tuple):
            safe_cfg[k] = list(v)
        else:
            safe_cfg[k] = v

    with open(
        os.path.join(result_dir, "run_config.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(safe_cfg, f, ensure_ascii=False, indent=2)



# ============================================================
# Optional Layer-2 calibration-pool prediction
# ============================================================
def predict_cal2_if_available(
    model,
    cfg
):
    x_path = cfg["cal2_x_path"]
    y_path = cfg["cal2_y_path"]

    # Nothing to do yet
    if x_path is None and y_path is None:
        print(
            "\nLayer-2 calibration pool paths are None; "
            "skip Cal-L2 prediction."
        )
        return None

    if x_path is None or y_path is None:
        raise ValueError(
            "cal2_x_path and cal2_y_path must either both be set "
            "or both be None."
        )

    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"Cal-L2 X file not found: {x_path}"
        )

    if not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Cal-L2 Y file not found: {y_path}"
        )

    X_cal2_df = read_numeric_csv_auto(
        x_path
    )

    Y_cal2_df = read_numeric_csv_auto(
        y_path
    )

    X_cal2 = X_cal2_df.to_numpy(
        dtype=np.float64
    )

    Y_cal2 = Y_cal2_df.iloc[:, :2].to_numpy(
        dtype=np.float64
    )

    check_xy(
        X_cal2,
        Y_cal2,
        name="Layer-2 calibration pool"
    )

    (
        mean,
        std_total,
        std_ep,
        std_al,
    ) = model.predict(
        X_cal2
    )

    result_dir = cfg["result_dir"]

    export_predictions(
        y_true=Y_cal2,
        mean=mean,
        std_total=std_total,
        std_ep=std_ep,
        std_al=std_al,
        sample_index=np.arange(1, len(Y_cal2) + 1),
        output_path=os.path.join(
            result_dir,
            "de_cal2_predictions.csv"
        ),
    )

    # Accuracy is only descriptive.
    evaluate_predictions(
        y_true=Y_cal2,
        y_pred=mean,
        result_dir=result_dir,
        prefix="cal2",
        std_total=std_total,
        std_ep=std_ep,
        std_al=std_al,
    )

    cal2_index = np.arange(1, len(Y_cal2) + 1)

    export_member_predictions(
        model, X_cal2, Y_cal2,
        cal2_index, result_dir, prefix="cal2"
    )

    plot_errorbar_parity(
        Y_cal2, mean, std_total,
        cal2_index, result_dir, prefix="cal2"
    )
    plot_sorted_prediction_band(
        Y_cal2, mean, std_total,
        cal2_index, result_dir
    )
    plot_uncertainty_vs_error(
        Y_cal2, mean, std_total,
        cal2_index, result_dir
    )
    plot_uncertainty_decomposition(
        std_total, std_ep, std_al,
        cal2_index, result_dir
    )
    plot_residuals(
        Y_cal2, mean,
        cal2_index, result_dir
    )

    return {
        "X": X_cal2,
        "Y": Y_cal2,
        "mean": mean,
        "std_total": std_total,
        "std_ep": std_ep,
        "std_al": std_al,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(
        f"Running script: "
        f"{os.path.abspath(__file__)}"
    )

    set_seed(
        CONFIG["random_seed"]
    )

    os.makedirs(
        CONFIG["result_dir"],
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read original data
    # --------------------------------------------------------
    print("\nLoading original data...")

    X_df = read_numeric_csv_auto(
        CONFIG["x_path"]
    )

    Y_df = read_numeric_csv_auto(
        CONFIG["y_path"]
    )

    if Y_df.shape[1] < 2:
        raise ValueError(
            "Y CSV must contain at least two columns: "
            "eta_head and eta_mid_max."
        )

    X = X_df.to_numpy(
        dtype=np.float64
    )

    Y = Y_df.iloc[:, :2].to_numpy(
        dtype=np.float64
    )

    check_xy(
        X,
        Y,
        name="Original dataset"
    )

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print("Y1 = eta_head")
    print("Y2 = eta_mid_max")

    # --------------------------------------------------------
    # Explicit sequential allocation
    # --------------------------------------------------------
    train_start = CONFIG["train_start"]
    train_end = CONFIG["train_end"]

    test_start = CONFIG["test_start"]
    test_end = CONFIG["test_end"]

    if len(X) < test_end:
        raise ValueError(
            f"Original dataset contains only {len(X)} samples, "
            f"but test_end={test_end}."
        )

    X_train = X[
        train_start:train_end
    ]

    Y_train = Y[
        train_start:train_end
    ]

    X_test = X[
        test_start:test_end
    ]

    Y_test = Y[
        test_start:test_end
    ]

    print("\n" + "=" * 72)
    print("Sequential data allocation")
    print("=" * 72)

    print(
        f"DE training    : "
        f"rows {train_start + 1}-{train_end} "
        f"(n={len(X_train)})"
    )

    print(
        f"Reserved Cal-L1: "
        f"rows {train_end + 1}-{test_start} "
        f"(n={test_start - train_end})"
    )

    print(
        f"Independent test: "
        f"rows {test_start + 1}-{test_end} "
        f"(n={len(X_test)})"
    )

    print(
        "\nThe reserved Cal-L1 samples are NOT used "
        "for Deep Ensemble training."
    )

    # --------------------------------------------------------
    # Train pure DE
    # --------------------------------------------------------
    model = DeepEnsembleETA(
        CONFIG
    )

    model.fit(
        X_train,
        Y_train
    )

    # --------------------------------------------------------
    # Training-set prediction and complete export
    # --------------------------------------------------------
    print(
        "\nPredicting the training set..."
    )

    (
        train_mean,
        train_std_total,
        train_std_ep,
        train_std_al,
    ) = model.predict(
        X_train
    )

    # 1-based original sample number: train_start+1 ... train_end
    train_original_index = np.arange(
        train_start + 1,
        train_end + 1
    )

    # Save complete aggregate DE predictions for TRAIN
    export_predictions(
        y_true=Y_train,
        mean=train_mean,
        std_total=train_std_total,
        std_ep=train_std_ep,
        std_al=train_std_al,
        sample_index=train_original_index,
        output_path=os.path.join(
            CONFIG["result_dir"],
            "de_train_predictions.csv"
        ),
    )

    # Save descriptive TRAIN metrics
    evaluate_predictions(
        y_true=Y_train,
        y_pred=train_mean,
        result_dir=CONFIG["result_dir"],
        prefix="train",
        std_total=train_std_total,
        std_ep=train_std_ep,
        std_al=train_std_al,
    )

    # Save member-level TRAIN predictions
    export_member_predictions(
        model,
        X_train,
        Y_train,
        train_original_index,
        CONFIG["result_dir"],
        prefix="train"
    )

    # --------------------------------------------------------
    # Independent test prediction
    # --------------------------------------------------------
    print(
        "\nPredicting the independent test set..."
    )

    (
        test_mean,
        test_std_total,
        test_std_ep,
        test_std_al,
    ) = model.predict(
        X_test
    )

    # 1-based original sample number: 901 ... 1000
    original_test_index = np.arange(
        test_start + 1,
        test_end + 1
    )

    export_predictions(
        y_true=Y_test,
        mean=test_mean,
        std_total=test_std_total,
        std_ep=test_std_ep,
        std_al=test_std_al,
        sample_index=original_test_index,
        output_path=os.path.join(
            CONFIG["result_dir"],
            "de_test_predictions.csv"
        ),
    )

    evaluate_predictions(
        y_true=Y_test,
        y_pred=test_mean,
        result_dir=CONFIG["result_dir"],
        prefix="test",
        std_total=test_std_total,
        std_ep=test_std_ep,
        std_al=test_std_al,
    )

    # --------------------------------------------------------
    # Comprehensive data / figure export for independent TEST
    # --------------------------------------------------------
    export_dataset_split(
        X_train, Y_train,
        X_test, Y_test,
        train_original_index,
        original_test_index,
        CONFIG["result_dir"]
    )

    export_training_history(
        model,
        CONFIG["result_dir"]
    )

    export_member_predictions(
        model,
        X_test,
        Y_test,
        original_test_index,
        CONFIG["result_dir"],
        prefix="test"
    )

    plot_errorbar_parity(
        Y_test,
        test_mean,
        test_std_total,
        original_test_index,
        CONFIG["result_dir"],
        prefix="test"
    )

    plot_sorted_prediction_band(
        Y_test,
        test_mean,
        test_std_total,
        original_test_index,
        CONFIG["result_dir"]
    )

    plot_uncertainty_vs_error(
        Y_test,
        test_mean,
        test_std_total,
        original_test_index,
        CONFIG["result_dir"]
    )

    plot_uncertainty_decomposition(
        test_std_total,
        test_std_ep,
        test_std_al,
        original_test_index,
        CONFIG["result_dir"]
    )

    plot_residuals(
        Y_test,
        test_mean,
        original_test_index,
        CONFIG["result_dir"]
    )

    export_run_config(
        CONFIG,
        CONFIG["result_dir"]
    )

    # --------------------------------------------------------
    # Optional additional Cal-L2 prediction
    # --------------------------------------------------------
    predict_cal2_if_available(
        model,
        CONFIG
    )

    # --------------------------------------------------------
    # Save DE model
    # --------------------------------------------------------
    model.save(
        CONFIG["model_path"]
    )

    print("\n" + "=" * 72)
    print("All done")
    print("=" * 72)

    print(
        f"Results directory: "
        f"{os.path.abspath(CONFIG['result_dir'])}"
    )

    print("Generated now:")
    for name in [
        "de_train_predictions.csv",
        "train_metrics.csv",
        "train_member_predictions.csv",
        "de_test_predictions.csv",
        "test_metrics.csv",
        "test_member_predictions.csv",
        "training_history_all_members.csv",
        "run_config.json",
        "fig_training_nll_all_members.png",
        "fig_errorbar_parity_eta_head.png",
        "fig_errorbar_parity_eta_mid_max.png",
        "fig_sorted_prediction_band_eta_head.png",
        "fig_sorted_prediction_band_eta_mid_max.png",
        "fig_uncertainty_vs_error_eta_head.png",
        "fig_uncertainty_vs_error_eta_mid_max.png",
        "fig_uncertainty_decomposition_eta_head.png",
        "fig_uncertainty_decomposition_eta_mid_max.png",
        "fig_residual_eta_head.png",
        "fig_residual_eta_mid_max.png",
        "data_split/train_data_used.csv",
        "data_split/test_data_used.csv",
        "fig_data_errorbar_parity/*.csv",
        "fig_data_sorted_prediction_band/*.csv",
        "fig_data_uncertainty_vs_error/*.csv",
        "fig_data_uncertainty_decomposition/*.csv",
        "fig_data_residuals/*.csv",
        CONFIG["model_path"],
    ]:
        print("  " + str(name))

    if (
        CONFIG["cal2_x_path"] is not None
        and CONFIG["cal2_y_path"] is not None
    ):
        print(
            "  de_cal2_predictions.csv"
        )
        print(
            "  cal2_metrics.csv"
        )