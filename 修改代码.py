import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
单阈值三区划分：基于最大轮对抬升的共形校准稳定性判别

新版改动：
1. 删除 -5 mm 下限，只保留最大轮对抬升阈值 h_lim = 20 mm；
2. q_lift 只基于最大抬升极值误差：
       score_i = |true_max_i - pred_max_i|
3. 使用有限样本修正的顺序统计量：
       k = ceil((n_cal + 1) * (1 - alpha))
       q_lift = 第 k 个排序后的 score
   不再使用 np.quantile 插值；
4. 不再拼接 max/min 误差；
5. 不再过滤所谓 crash 样本，以避免改变校准分布；
6. 测试集真实稳定/失稳直接根据 true_max 与 20 mm 判断，
   不再依赖旧 label；
7. 增加校准稳定区误判率的 95% 单侧二项置信上界。
"""

# ============================================================
# 绘图设置
# ============================================================
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "calib_path": r"E:\Python_Document\RESS\1new_data\calib_lift.csv",
    "test_path": r"E:\Python_Document\RESS\1new_data\test_lift_modified.csv",
    "save_dir": "results_zone_single_threshold",

    "h_lim": 20.0,       # mm，最大轮对抬升阈值
    "alpha": 0.05,       # 目标覆盖率 95%
    "confidence": 0.95,  # 二项置信上界
}


# ============================================================
# 单侧精确二项分布置信上界
# ============================================================
def one_sided_binomial_upper(x, n, confidence=0.95):
    """
    二项比例的单侧精确置信上界。

    x=0 时：
        upper = 1 - (1-confidence)^(1/n)

    例如：
        x=0, n=49, confidence=0.95
        upper ≈ 0.0593
    """
    if n <= 0:
        return np.nan

    if x == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)

    if x >= n:
        return 1.0

    try:
        from scipy.stats import beta
        return float(beta.ppf(confidence, x + 1, n - x))
    except ImportError:
        return np.nan


# ============================================================
# 第一步：计算 q_lift
# ============================================================
def compute_q_lift(calib, cfg):
    """
    非符合度分数：
        s_i = |true_max_i - pred_max_i|

    有限样本修正：
        k = ceil((n_cal + 1) * (1-alpha))

    q_lift：
        排序后第 k 个非符合度分数
    """
    alpha = cfg["alpha"]

    required_cols = ["pred_max", "true_max"]
    missing = [c for c in required_cols if c not in calib.columns]
    if missing:
        raise ValueError(
            f"calib_lift.csv 缺少必要列：{missing}，至少需要 {required_cols}"
        )

    pred_max = calib["pred_max"].to_numpy(dtype=float)
    true_max = calib["true_max"].to_numpy(dtype=float)

    valid_mask = np.isfinite(pred_max) & np.isfinite(true_max)
    if not np.all(valid_mask):
        print(
            f"警告：校准集中发现 {(~valid_mask).sum()} 个 NaN/Inf 样本，"
            f"这些无效样本将被删除。"
        )

    pred_max = pred_max[valid_mask]
    true_max = true_max[valid_mask]

    # 每个校准样本一个 score
    scores = np.abs(true_max - pred_max)

    n_cal = len(scores)
    if n_cal == 0:
        raise ValueError("校准集中没有有效样本。")

    # 有限样本 conformal 顺序统计量位置
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))

    if k > n_cal:
        raise ValueError(
            f"当前 n_cal={n_cal}, alpha={alpha} 时得到 k={k} > n_cal。"
            f"校准样本数不足以支持目标覆盖率 {1-alpha:.1%}。"
        )

    sorted_scores = np.sort(scores)

    # 第 k 个顺序统计量，Python 下标从 0 开始
    q_lift = float(sorted_scores[k - 1])

    print("\n" + "=" * 60)
    print("Conformal calibration")
    print("=" * 60)
    print(f"校准样本数 n_cal = {n_cal}")
    print(f"alpha            = {alpha}")
    print(f"目标覆盖率        = {1-alpha:.1%}")
    print(
        f"k = ceil((n_cal+1)*(1-alpha)) "
        f"= ceil(({n_cal}+1)*{1-alpha:.2f}) = {k}"
    )
    print(f"q_lift = s_({k}) = {q_lift:.6f} mm")

    print("\n最大抬升极值绝对误差统计：")
    print(f"  Mean = {scores.mean():.6f} mm")
    print(f"  RMSE = {np.sqrt(np.mean(scores ** 2)):.6f} mm")
    print(f"  Max  = {scores.max():.6f} mm")
    print(f"  Q90  = {np.quantile(scores, 0.90):.6f} mm")
    print(f"  Q95（仅描述性统计） = {np.quantile(scores, 0.95):.6f} mm")
    print(f"  Conformal q_lift   = {q_lift:.6f} mm")

    calib_used = pd.DataFrame({
        "pred_max": pred_max,
        "true_max": true_max,
        "score": scores,
    })

    return q_lift, scores, n_cal, k, calib_used


# ============================================================
# 第二步：三区划分
# ============================================================
def three_zone_classify(test, q_lift, cfg):
    """
    可信稳定：
        pred_max + q_lift <= h_lim

    可信失稳：
        pred_max - q_lift > h_lim

    待验证：
        其余
    """
    h_lim = cfg["h_lim"]

    required_cols = ["pred_max", "true_max"]
    missing = [c for c in required_cols if c not in test.columns]
    if missing:
        raise ValueError(
            f"test_lift.csv 缺少必要列：{missing}，至少需要 {required_cols}"
        )

    pred_max = test["pred_max"].to_numpy(dtype=float)

    if not np.all(np.isfinite(pred_max)):
        raise ValueError("测试集 pred_max 中存在 NaN/Inf，请先清理。")

    lower = pred_max - q_lift
    upper = pred_max + q_lift

    zone_stable = upper <= h_lim
    zone_unstable = lower > h_lim
    zone_uncertain = ~(zone_stable | zone_unstable)

    reason = np.empty(len(test), dtype=object)
    reason[zone_stable] = "校准稳定"
    reason[zone_unstable] = "校准失稳"
    reason[zone_uncertain] = "待验证"

    return zone_stable, zone_unstable, zone_uncertain, lower, upper, reason


# ============================================================
# 第三步：验证三区划分
# ============================================================
def validate_zones(
    test,
    zone_stable,
    zone_unstable,
    zone_uncertain,
    cfg
):
    """
    新版真实稳定/失稳定义：
        true stable   : true_max <= h_lim
        true unstable : true_max >  h_lim
    """
    h_lim = cfg["h_lim"]
    confidence = cfg["confidence"]

    true_max = test["true_max"].to_numpy(dtype=float)
    true_fail = true_max > h_lim
    true_stable = ~true_fail

    n = len(test)

    n_stable = int(zone_stable.sum())
    n_unstable = int(zone_unstable.sum())
    n_uncertain = int(zone_uncertain.sum())

    # 校准稳定区中的真实失稳：false acceptance
    false_accept_count = int(true_fail[zone_stable].sum())
    false_accept_rate = (
        false_accept_count / n_stable if n_stable > 0 else np.nan
    )

    # 校准失稳区中的真实稳定：保守误判
    conservative_error_count = int(true_stable[zone_unstable].sum())
    conservative_error_rate = (
        conservative_error_count / n_unstable
        if n_unstable > 0 else np.nan
    )

    # 待验证区中的真实失稳
    uncertain_fail_count = int(true_fail[zone_uncertain].sum())
    uncertain_fail_rate = (
        uncertain_fail_count / n_uncertain
        if n_uncertain > 0 else np.nan
    )

    false_accept_upper = one_sided_binomial_upper(
        false_accept_count,
        n_stable,
        confidence=confidence
    )

    print("\n" + "=" * 60)
    print("三区划分结果")
    print("=" * 60)
    print(
        f"校准稳定区 : {n_stable:3d} 样本 "
        f"({n_stable / n * 100:.1f}%)"
    )
    print(
        f"待验证区   : {n_uncertain:3d} 样本 "
        f"({n_uncertain / n * 100:.1f}%)"
    )
    print(
        f"校准失稳区 : {n_unstable:3d} 样本 "
        f"({n_unstable / n * 100:.1f}%)"
    )

    print("\n基于 FE true_max 的验证：")

    if n_stable > 0:
        print(
            f"校准稳定区中的真实失稳："
            f"{false_accept_count}/{n_stable}"
        )
        print(
            f"观测 false-acceptance rate = "
            f"{false_accept_rate:.4f} "
            f"({false_accept_rate*100:.2f}%)"
        )

        if np.isfinite(false_accept_upper):
            print(
                f"{confidence:.0%} 单侧精确二项置信上界 = "
                f"{false_accept_upper:.4f} "
                f"({false_accept_upper*100:.2f}%)"
            )

    if n_unstable > 0:
        print(
            f"校准失稳区中的真实稳定："
            f"{conservative_error_count}/{n_unstable}"
        )
        print(
            f"保守误判率 = "
            f"{conservative_error_rate:.4f} "
            f"({conservative_error_rate*100:.2f}%)"
        )

    if n_uncertain > 0:
        print(
            f"待验证区真实失稳："
            f"{uncertain_fail_count}/{n_uncertain} "
            f"({uncertain_fail_rate*100:.2f}%)"
        )

    return {
        "n_total": n,
        "n_stable": n_stable,
        "n_uncertain": n_uncertain,
        "n_unstable": n_unstable,

        "false_accept_count": false_accept_count,
        "false_accept_rate": false_accept_rate,
        "false_accept_upper_95": false_accept_upper,

        "conservative_error_count": conservative_error_count,
        "conservative_error_rate": conservative_error_rate,

        "uncertain_fail_count": uncertain_fail_count,
        "uncertain_fail_rate": uncertain_fail_rate,
    }


# ============================================================
# 图1：校准集误差分布
# ============================================================
def plot_error_dist(scores, q_lift, save_dir):
    """
    图1：校准集误差分布
    - 纵坐标：Density
    - 固定分箱宽度：0.25 mm
    - 保存直方图绘图数据
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # ========================================================
    # 固定分箱宽度为 0.25 mm
    # 0~1 mm 正好对应 4 个柱子
    # ========================================================
    bin_width = 0.2

    max_score = np.max(scores)

    bin_edges = np.arange(
        0,
        np.ceil(max_score / bin_width) * bin_width + bin_width,
        bin_width
    )

    counts, bin_edges, _ = ax.hist(
        scores,
        bins=bin_edges,
        density=True,
        alpha=0.75,
        edgecolor="white",
        label="Histogram"
    )

    mean_score = float(np.mean(scores))

    # q_lift
    ax.axvline(
        q_lift,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=rf"$q_{{lift}}={q_lift:.3f}$ mm"
    )

    # Mean
    ax.axvline(
        mean_score,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean={mean_score:.3f} mm"
    )

    ax.set_xlabel("Absolute extrema error (mm)")
    ax.set_ylabel("Density")

    ax.legend(frameon=False)
    ax.tick_params(
        direction="in",
        top=False,
        right=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "fig1_error_dist.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 保存 Fig.1 绘图数据
    # ========================================================
    hist_df = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "bin_center": (
            bin_edges[:-1] + bin_edges[1:]
        ) / 2,
        "density": counts
    })

    hist_df.to_csv(
        os.path.join(
            save_dir,
            "data_fig1_error_dist.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # 保存两条竖线数据
    reference_df = pd.DataFrame({
        "parameter": ["Mean", "q_lift"],
        "value": [mean_score, q_lift]
    })

    reference_df.to_csv(
        os.path.join(
            save_dir,
            "data_fig1_error_dist_reference.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 图2：预测最大抬升 vs FE真实最大抬升
# ============================================================
def plot_pred_vs_true(
    test,
    zone_stable,
    zone_unstable,
    zone_uncertain,
    q_lift,
    cfg,
    save_dir
):
    h_lim = cfg["h_lim"]

    pred_max = test["pred_max"].to_numpy(dtype=float)
    true_max = test["true_max"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        true_max[zone_stable],
        pred_max[zone_stable],
        s=35,
        alpha=0.8,
        label=f"Calibrated stable (n={zone_stable.sum()})"
    )
    ax.scatter(
        true_max[zone_uncertain],
        pred_max[zone_uncertain],
        s=45,
        alpha=0.8,
        marker="^",
        label=f"To-be-verified (n={zone_uncertain.sum()})"
    )
    ax.scatter(
        true_max[zone_unstable],
        pred_max[zone_unstable],
        s=35,
        alpha=0.8,
        label=f"Calibrated unstable (n={zone_unstable.sum()})"
    )

    all_values = np.concatenate([true_max, pred_max])
    lo = np.nanmin(all_values)
    hi = np.nanmax(all_values)
    pad = max((hi - lo) * 0.05, 1.0)
    xy_min = lo - pad
    xy_max = hi + pad

    ax.plot(
        [xy_min, xy_max],
        [xy_min, xy_max],
        linestyle="--",
        linewidth=1.5,
        label="y=x"
    )

    ax.axhline(
        h_lim,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {h_lim:g} mm"
    )
    ax.axhline(
        h_lim - q_lift,
        linestyle=":",
        linewidth=1.2
    )
    ax.axhline(
        h_lim + q_lift,
        linestyle=":",
        linewidth=1.2,
        label=rf"Threshold $\pm q_{{lift}}$"
    )

    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    ax.set_xlabel("FE maximum wheelset lift (mm)")
    ax.set_ylabel("Predicted maximum wheelset lift (mm)")
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=False, right=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "fig2_pred_vs_true.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 图3：各区真实稳定/失稳数量
# ============================================================
def plot_zone_validation(
    test,
    zone_stable,
    zone_unstable,
    zone_uncertain,
    cfg,
    save_dir
):
    h_lim = cfg["h_lim"]
    true_fail = test["true_max"].to_numpy(dtype=float) > h_lim

    zones = {
        "Calibrated stable": zone_stable,
        "To-be-verified": zone_uncertain,
        "Calibrated unstable": zone_unstable,
    }

    zone_names = list(zones.keys())
    n_true_stable = [
        int((~true_fail[m]).sum()) for m in zones.values()
    ]
    n_true_unstable = [
        int(true_fail[m].sum()) for m in zones.values()
    ]

    x = np.arange(len(zone_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(
        x - w / 2,
        n_true_stable,
        w,
        label="FE stable"
    )
    bars2 = ax.bar(
        x + w / 2,
        n_true_unstable,
        w,
        label="FE unstable"
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{int(h)}",
                    ha="center",
                    va="bottom"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(zone_names)
    ax.set_ylabel("Number of samples")
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=False, right=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "fig3_zone_validation.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 图4：按 pred_max 排序展示三区划分
# ============================================================
def plot_zone_sorted(
    test,
    zone_stable,
    zone_unstable,
    zone_uncertain,
    q_lift,
    cfg,
    save_dir
):
    h_lim = cfg["h_lim"]

    pred_max = test["pred_max"].to_numpy(dtype=float)
    true_max = test["true_max"].to_numpy(dtype=float)

    # ========================================================
    # 按预测最大抬升从小到大排序
    # ========================================================
    order = np.argsort(pred_max)

    pred_sorted = pred_max[order]
    true_sorted = true_max[order]

    stable_sorted = zone_stable[order]
    unstable_sorted = zone_unstable[order]
    uncertain_sorted = zone_uncertain[order]

    # 排序后的横坐标
    x = np.arange(len(pred_sorted))

    # 共形上下界
    lower_sorted = pred_sorted - q_lift
    upper_sorted = pred_sorted + q_lift

    # ========================================================
    # 分类标签
    # ========================================================
    zone_sorted = np.where(
        stable_sorted,
        "calibrated_stable",
        np.where(
            unstable_sorted,
            "calibrated_unstable",
            "to_be_verified"
        )
    )

    # ========================================================
    # 保存 Fig.4 绘图数据
    # ========================================================
    fig4_data = pd.DataFrame({
        # 原始 test.csv 中的行号，从 1 开始
        "original_sample_index": order + 1,

        # 图中的横坐标，从 0 开始
        "sorted_index": x,

        # FE 真实最大轮对抬升
        "true_max": true_sorted,

        # PA-DA-RNN 预测最大轮对抬升
        "pred_max": pred_sorted,

        # 共形预测区间
        "pred_interval_lower": lower_sorted,
        "pred_interval_upper": upper_sorted,

        # 三区分类
        "zone": zone_sorted,

        # 画图参考参数
        "h_lim": h_lim,
        "q_lift": q_lift
    })

    fig4_data.to_csv(
        os.path.join(
            save_dir,
            "data_fig4_zone_sorted.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 绘图
    # ========================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # 共形预测区间
    ax.fill_between(
        x,
        lower_sorted,
        upper_sorted,
        alpha=0.1,
        label=rf"$\hat h_{{max}}\pm q_{{lift}}$"
    )

    # 校准稳定区
    ax.scatter(
        x[stable_sorted],
        pred_sorted[stable_sorted],
        s=25,
        label="Calibrated stable"
    )

    # 待验证区
    ax.scatter(
        x[uncertain_sorted],
        pred_sorted[uncertain_sorted],
        s=35,
        marker="^",
        label="To-be-verified"
    )

    # 校准失稳区
    ax.scatter(
        x[unstable_sorted],
        pred_sorted[unstable_sorted],
        s=25,
        label="Calibrated unstable"
    )

    # 20 mm 阈值
    ax.axhline(
        h_lim,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {h_lim:g} mm"
    )

    ax.set_xlabel(
        "Samples sorted by predicted maximum lift"
    )

    ax.set_ylabel(
        "Maximum wheelset lift (mm)"
    )

    ax.legend(
        frameon=False
    )

    ax.tick_params(
        direction="in",
        top=False,
        right=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "fig4_zone_sorted.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 保存校准分数
# ============================================================
def save_calibration_scores(
    calib_used,
    q_lift,
    k,
    save_dir
):
    df = calib_used.copy()

    order = np.argsort(df["score"].to_numpy())
    rank = np.empty(len(df), dtype=int)
    rank[order] = np.arange(1, len(df) + 1)

    df["order_rank"] = rank
    df["is_selected_q"] = df["order_rank"] == k

    df.to_csv(
        os.path.join(save_dir, "calibration_scores.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([{
        "selected_order_k": k,
        "q_lift": q_lift
    }]).to_csv(
        os.path.join(save_dir, "conformal_quantile_detail.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 保存测试结果
# ============================================================
def save_results(
    test,
    zone_stable,
    zone_unstable,
    zone_uncertain,
    lower,
    upper,
    reason,
    q_lift,
    n_cal,
    k,
    val_result,
    cfg,
    save_dir
):
    h_lim = cfg["h_lim"]

    zone_label = np.where(
        zone_stable,
        "calibrated_stable",
        np.where(
            zone_unstable,
            "calibrated_unstable",
            "to_be_verified"
        )
    )

    df_out = test.copy()

    # 新版真实标签，只依据 true_max 与 20 mm
    df_out["true_status_new"] = np.where(
        df_out["true_max"].to_numpy(dtype=float) <= h_lim,
        "stable",
        "unstable"
    )

    df_out["zone"] = zone_label
    df_out["reason"] = reason
    df_out["q_lift"] = q_lift
    df_out["pred_interval_lower"] = lower
    df_out["pred_interval_upper"] = upper

    true_fail = df_out["true_max"].to_numpy(dtype=float) > h_lim

    misjudge = np.array(["none"] * len(df_out), dtype=object)
    misjudge[zone_stable & true_fail] = "false_acceptance"
    misjudge[zone_unstable & (~true_fail)] = "conservative_error"
    df_out["misjudge"] = misjudge

    df_out.to_csv(
        os.path.join(save_dir, "test_zone_result.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "alpha": cfg["alpha"],
        "coverage_target": 1 - cfg["alpha"],
        "n_cal": n_cal,
        "order_statistic_k": k,
        "q_lift": q_lift,
        "h_lim": h_lim,

        "n_total_test": val_result["n_total"],
        "n_calibrated_stable": val_result["n_stable"],
        "n_to_be_verified": val_result["n_uncertain"],
        "n_calibrated_unstable": val_result["n_unstable"],

        "false_accept_count": val_result["false_accept_count"],
        "false_accept_rate": val_result["false_accept_rate"],
        "false_accept_upper_95": val_result["false_accept_upper_95"],

        "conservative_error_count": val_result["conservative_error_count"],
        "conservative_error_rate": val_result["conservative_error_rate"],

        "uncertain_fail_count": val_result["uncertain_fail_count"],
        "uncertain_fail_rate": val_result["uncertain_fail_rate"],
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(save_dir, "q_lift_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    cfg = CONFIG
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    calib = pd.read_csv(cfg["calib_path"])
    test = pd.read_csv(cfg["test_path"])

    # 兼容旧拼写，但新版不再依赖 label
    test = test.rename(columns={"lable": "label"})

    print("=" * 60)
    print("第一步：计算有限样本修正后的 q_lift")
    print("=" * 60)

    q_lift, scores, n_cal, k, calib_used = compute_q_lift(
        calib,
        cfg
    )

    print("\n" + "=" * 60)
    print("第二步：基于 20 mm 单阈值进行三区划分")
    print("=" * 60)

    (
        zone_stable,
        zone_unstable,
        zone_uncertain,
        lower,
        upper,
        reason,
    ) = three_zone_classify(
        test,
        q_lift,
        cfg
    )

    print("\n" + "=" * 60)
    print("第三步：使用 FE true_max 验证三区结果")
    print("=" * 60)

    val_result = validate_zones(
        test,
        zone_stable,
        zone_unstable,
        zone_uncertain,
        cfg
    )

    print("\n" + "=" * 60)
    print("第四步：保存结果与图表")
    print("=" * 60)

    save_calibration_scores(
        calib_used,
        q_lift,
        k,
        save_dir
    )

    plot_error_dist(
        scores,
        q_lift,
        save_dir
    )

    plot_pred_vs_true(
        test,
        zone_stable,
        zone_unstable,
        zone_uncertain,
        q_lift,
        cfg,
        save_dir
    )

    plot_zone_validation(
        test,
        zone_stable,
        zone_unstable,
        zone_uncertain,
        cfg,
        save_dir
    )

    plot_zone_sorted(
        test,
        zone_stable,
        zone_unstable,
        zone_uncertain,
        q_lift,
        cfg,
        save_dir
    )

    save_results(
        test,
        zone_stable,
        zone_unstable,
        zone_uncertain,
        lower,
        upper,
        reason,
        q_lift,
        n_cal,
        k,
        val_result,
        cfg,
        save_dir
    )

    print("\n" + "=" * 60)
    print("全部完成")
    print("=" * 60)
    print(f"q_lift              = {q_lift:.6f} mm")
    print(f"n_cal               = {n_cal}")
    print(f"k                   = {k}")
    print(f"最大抬升阈值         = {cfg['h_lim']} mm")
    print(f"校准稳定区           = {val_result['n_stable']} 样本")
    print(f"待验证区             = {val_result['n_uncertain']} 样本")
    print(f"校准失稳区           = {val_result['n_unstable']} 样本")

    if val_result["n_stable"] > 0:
        print(
            f"校准稳定区观测误判率  = "
            f"{val_result['false_accept_rate']*100:.2f}%"
        )

        if np.isfinite(val_result["false_accept_upper_95"]):
            print(
                f"误判率95%单侧上置信界 = "
                f"{val_result['false_accept_upper_95']*100:.2f}%"
            )

    print(f"\n结果目录：{save_dir}")
    print("  calibration_scores.csv")
    print("  conformal_quantile_detail.csv")
    print("  q_lift_summary.csv")
    print("  test_zone_result.csv")
    print("  fig1_error_dist.png")
    print("  data_fig1_error_dist.csv")
    print("  data_fig1_error_dist_reference.csv")
    print("  fig2_pred_vs_true.png")
    print("  fig3_zone_validation.png")
    print("  fig4_zone_sorted.png")
    print("=" * 60)