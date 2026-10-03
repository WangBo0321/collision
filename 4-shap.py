import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# Plot style
# ============================================================
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

DPI = 300
FONT_TICK = 14
FONT_LABEL = 16


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # 原始输入数据
    "x_path": r"D.csv",

    # 你训练代码保存的 Deep Ensemble
    "model_path": r"deep_ensemble_eta.pkl",

    # 输出目录
    "result_dir": r"results_shap_eta",

    # 与原训练代码保持一致
    "train_start": 0,
    "train_end": 1000,

    "test_start": 1000,
    "test_end": 1100,

    # --------------------------------------------------------
    # 推荐：这里填入 55 个 calibrated-stable 测试样本的原始 sample_index
    #
    # CSV 只需要一列：
    # sample_index
    # 1042
    # 1044
    # ...
    #
    # sample_index 必须与你原程序输出的 sample_index 一致（从1开始）
    #
    # 如果暂时设为 None：
    # 则默认解释全部 100 个独立测试样本。
    # --------------------------------------------------------
    "stable_index_path": None,
    # 例如：
    # "stable_index_path": r"stable_test_indices.csv",

    # SHAP background，从训练数据中随机抽取
    "background_size": 100,

    # permutation SHAP 的计算次数
    # 16个特征时，理论最低为 2*16+1 = 33
    # 这里增加重复次数提高稳定性
    "n_permutations": 10,

    "random_seed": 42,
}


# ============================================================
# 特征名称
# 必须与 D.csv 的 16 列顺序完全一致
# ============================================================
FEATURE_NAMES = [
    "F_ac",
    "F_ea",
    "F_hc",
    "S_ac",
    "S_ea",
    "S_hc",
    "F_s1",
    "F_s2",
    "F_s3",
    "F_s4",
    "S_s1",
    "S_s2",
    "S_s3",
    "S_s4",
    "H_off",
    "V_off",
]

OUTPUT_NAMES = [
    "eta_head",
    "eta_mid_max",
]


# ============================================================
# Seed
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CSV reader
# ============================================================
def read_numeric_csv_auto(path):
    raw = pd.read_csv(path, header=None)

    num = raw.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 自动识别表头
    if (
        len(num) >= 2
        and num.iloc[0].isna().any()
        and num.iloc[1:].notna().all().all()
    ):
        num = num.iloc[1:].reset_index(drop=True)

    if num.isna().any().any():
        r, c = np.argwhere(
            num.isna().to_numpy()
        )[0]

        raise ValueError(
            f"{path} contains non-numeric/missing value "
            f"at row {r + 1}, col {c + 1}"
        )

    return num.astype(np.float64)


# ============================================================
# 必须保留与训练代码一致的类定义
# 否则 pickle.load 可能找不到类
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

        self.mean_head = nn.Linear(
            prev,
            out_dim
        )

        self.log_var_head = nn.Linear(
            prev,
            out_dim
        )

    def forward(self, x):
        feat = self.backbone(x)

        mean = self.mean_head(feat)

        log_var = torch.clamp(
            self.log_var_head(feat),
            -10.0,
            10.0
        )

        return mean, log_var


class DeepEnsembleETA:
    def __init__(self, cfg=None):
        self.cfg = cfg

        self.members = []
        self.training_histories = []

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        self.out_dim = None
        self.device = "cpu"

    def predict(self, X):

        if len(self.members) == 0:
            raise RuntimeError(
                "Deep Ensemble has no members."
            )

        X = np.asarray(
            X,
            dtype=np.float64
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
                    mu_sc.detach().cpu().numpy()
                )

                all_vars.append(
                    np.exp(
                        log_var_sc.detach().cpu().numpy()
                    )
                )

        all_means = np.asarray(all_means)
        all_vars = np.asarray(all_vars)

        mean_sc = all_means.mean(axis=0)

        var_ep_sc = all_means.var(axis=0)
        var_al_sc = all_vars.mean(axis=0)

        var_total_sc = (
            var_ep_sc
            + var_al_sc
        )

        std_ep_sc = np.sqrt(
            np.maximum(
                var_ep_sc,
                0.0
            )
        )

        std_al_sc = np.sqrt(
            np.maximum(
                var_al_sc,
                0.0
            )
        )

        std_total_sc = np.sqrt(
            np.maximum(
                var_total_sc,
                0.0
            )
        )

        mean = self.scaler_Y.inverse_transform(
            mean_sc
        )

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
            std_al
        )


# ============================================================
# Load model
# ============================================================
def load_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    with open(path, "rb") as f:
        model = pickle.load(f)

    # SHAP不需要GPU，CPU更稳定
    model.device = "cpu"

    for member in model.members:
        member.cpu()
        member.eval()

    print(
        f"Model loaded: "
        f"{os.path.abspath(path)}"
    )

    print(
        f"Ensemble members: "
        f"{len(model.members)}"
    )

    return model


# ============================================================
# Select samples for SHAP
# ============================================================
def prepare_data(cfg):

    X_df = read_numeric_csv_auto(
        cfg["x_path"]
    )

    X = X_df.to_numpy(
        dtype=np.float64
    )

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X has {X.shape[1]} columns, "
            f"but FEATURE_NAMES has {len(FEATURE_NAMES)}."
        )

    # -------------------------
    # Training background
    # -------------------------
    X_train = X[
        cfg["train_start"]:
        cfg["train_end"]
    ]

    rng = np.random.default_rng(
        cfg["random_seed"]
    )

    n_bg = min(
        cfg["background_size"],
        len(X_train)
    )

    bg_index = rng.choice(
        len(X_train),
        size=n_bg,
        replace=False
    )

    X_background = X_train[
        bg_index
    ]

    # -------------------------
    # SHAP解释对象
    # -------------------------
    stable_index_path = cfg[
        "stable_index_path"
    ]

    if stable_index_path is None:

        print(
            "\nWARNING:"
        )
        print(
            "stable_index_path = None"
        )
        print(
            "SHAP will explain all 100 test samples."
        )

        X_explain = X[
            cfg["test_start"]:
            cfg["test_end"]
        ]

        sample_index = np.arange(
            cfg["test_start"] + 1,
            cfg["test_end"] + 1
        )

    else:

        idx_df = pd.read_csv(
            stable_index_path
        )

        if "sample_index" not in idx_df.columns:
            raise ValueError(
                "stable_index_path must contain "
                "a column named 'sample_index'."
            )

        sample_index = (
            idx_df["sample_index"]
            .astype(int)
            .to_numpy()
        )

        # 原程序 sample_index 是1-based
        zero_based = (
            sample_index - 1
        )

        if (
            zero_based.min() < 0
            or zero_based.max() >= len(X)
        ):
            raise ValueError(
                "sample_index exceeds X range."
            )

        X_explain = X[
            zero_based
        ]

        print(
            f"\nUsing calibrated-stable samples "
            f"for SHAP: n={len(X_explain)}"
        )

    print(
        f"Background samples : {len(X_background)}"
    )

    print(
        f"Explained samples  : {len(X_explain)}"
    )

    return (
        X_background,
        X_explain,
        sample_index
    )


# ============================================================
# Prediction wrapper
# SHAP解释的是 Deep Ensemble predictive mean
# ============================================================
class DEOutputWrapper:

    def __init__(
        self,
        model,
        output_index
    ):
        self.model = model
        self.output_index = output_index

    def __call__(self, X):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        mean, _, _, _ = (
            self.model.predict(X)
        )

        # 返回单个输出
        return mean[
            :,
            self.output_index
        ]


# ============================================================
# SHAP analysis
# ============================================================
def run_shap_for_output(
    model,
    X_background,
    X_explain,
    sample_index,
    output_index,
    output_name,
    cfg
):

    print(
        "\n" + "=" * 72
    )

    print(
        f"SHAP analysis: {output_name}"
    )

    print(
        "=" * 72
    )

    predict_function = DEOutputWrapper(
        model,
        output_index
    )

    # --------------------------------------------------------
    # Permutation SHAP
    # 对 Deep Ensemble 这种黑箱模型比较稳妥
    # --------------------------------------------------------
    masker = shap.maskers.Independent(
        X_background
    )

    explainer = shap.Explainer(
        predict_function,
        masker,
        algorithm="permutation",
        feature_names=FEATURE_NAMES,
    )

    n_features = X_explain.shape[1]

    # 每轮 permutation 至少需要 2*p+1 次
    max_evals = (
        (2 * n_features + 1)
        * cfg["n_permutations"]
    )

    print(
        f"max_evals = {max_evals}"
    )

    shap_values = explainer(
        X_explain,
        max_evals=max_evals
    )

    result_dir = cfg[
        "result_dir"
    ]

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 保存逐样本 SHAP
    # --------------------------------------------------------
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=FEATURE_NAMES
    )

    shap_df.insert(
        0,
        "sample_index",
        sample_index
    )

    shap_df.to_csv(
        os.path.join(
            result_dir,
            f"shap_values_{output_name}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 全局重要性
    # mean(|SHAP|)
    # --------------------------------------------------------
    mean_abs_shap = np.mean(
        np.abs(
            shap_values.values
        ),
        axis=0
    )

    mean_shap = np.mean(
        shap_values.values,
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_SHAP": mean_abs_shap,
        "mean_SHAP": mean_shap,
    })

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df[
        "rank"
    ] = np.arange(
        1,
        len(importance_df) + 1
    )

    importance_df.to_csv(
        os.path.join(
            result_dir,
            f"shap_importance_{output_name}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nFeature importance:"
    )

    print(
        importance_df[
            [
                "rank",
                "feature",
                "mean_abs_SHAP"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Figure 1: Beeswarm
    # 同时看“重要性 + 参数高低值的影响方向”
    # ========================================================
    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=len(FEATURE_NAMES),
        show=False
    )

    fig = plt.gcf()

    fig.set_size_inches(
        8,
        6
    )

    ax = plt.gca()

    ax.tick_params(
        axis="both",
        labelsize=FONT_TICK
    )

    ax.set_xlabel(
        "SHAP value",
        fontsize=FONT_LABEL
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            f"shap_beeswarm_{output_name}.tif"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # 保存完整 beeswarm 绘图数据
    # 包括：sample_index + 原始特征值 + SHAP值
    # ============================================================
    beeswarm_df = pd.DataFrame({
        "sample_index": sample_index
    })

    for i, feature in enumerate(FEATURE_NAMES):
        beeswarm_df[f"{feature}_value"] = X_explain[:, i]
        beeswarm_df[f"{feature}_SHAP"] = shap_values.values[:, i]

    beeswarm_df.to_csv(
        os.path.join(
            result_dir,
            f"shap_beeswarm_data_{output_name}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )
    # ========================================================
    # Figure 2: Global importance bar
    # ========================================================
    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=len(FEATURE_NAMES),
        show=False
    )

    fig = plt.gcf()

    fig.set_size_inches(
        8,
        6
    )

    ax = plt.gca()

    ax.tick_params(
        axis="both",
        labelsize=FONT_TICK
    )

    ax.set_xlabel(
        "Mean |SHAP value|",
        fontsize=FONT_LABEL
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            f"shap_importance_{output_name}.tif"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    return (
        shap_values,
        importance_df
    )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    set_seed(
        CONFIG["random_seed"]
    )

    os.makedirs(
        CONFIG["result_dir"],
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load trained Deep Ensemble
    # --------------------------------------------------------
    model = load_model(
        CONFIG["model_path"]
    )

    # --------------------------------------------------------
    # Prepare background + samples to explain
    # --------------------------------------------------------
    (
        X_background,
        X_explain,
        sample_index
    ) = prepare_data(
        CONFIG
    )

    # --------------------------------------------------------
    # eta_head
    # --------------------------------------------------------
    shap_head, imp_head = (
        run_shap_for_output(
            model=model,
            X_background=X_background,
            X_explain=X_explain,
            sample_index=sample_index,
            output_index=0,
            output_name="eta_head",
            cfg=CONFIG
        )
    )

    # --------------------------------------------------------
    # eta_mid_max
    # --------------------------------------------------------
    shap_mid, imp_mid = (
        run_shap_for_output(
            model=model,
            X_background=X_background,
            X_explain=X_explain,
            sample_index=sample_index,
            output_index=1,
            output_name="eta_mid_max",
            cfg=CONFIG
        )
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SHAP analysis completed."
    )

    print(
        "=" * 72
    )

    print(
        f"Results directory: "
        f"{os.path.abspath(CONFIG['result_dir'])}"
    )

    print("\nGenerated files:")

    for f in [
        "shap_beeswarm_eta_head.tif",
        "shap_beeswarm_eta_mid_max.tif",
        "shap_importance_eta_head.tif",
        "shap_importance_eta_mid_max.tif",
        "shap_values_eta_head.csv",
        "shap_values_eta_mid_max.csv",
        "shap_importance_eta_head.csv",
        "shap_importance_eta_mid_max.csv",
    ]:
        print("  " + f)
