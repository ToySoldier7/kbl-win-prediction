from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


FEATURE_LABELS = {
    "points_per40": "득점(40분)",
    "two_pt_pct": "2점 성공률",
    "three_pt_pct": "3점 성공률",
    "free_throw_pct": "자유투 성공률",
    "off_rebounds_per40": "공격리바운드(40분)",
    "def_rebounds_per40": "수비리바운드(40분)",
    "assists_per40": "어시스트(40분)",
    "steals_per40": "스틸(40분)",
    "blocks_per40": "블록(40분)",
    "turnovers_per40": "턴오버(40분)",
    "fouls_per40": "파울(40분)",
    "efg_pct": "eFG%",
    "tov_pct": "TOV%",
    "oreb_pct": "ORB%",
    "fta_rate": "FTA Rate",
    "offensive_rating": "공격 효율",
    "defensive_rating": "수비 효율",
    "pace": "경기 템포",
}

ANALYSIS_FEATURES = list(FEATURE_LABELS)
WIN_COLOR = "#2468A2"
LOSS_COLOR = "#E69F00"
NEUTRAL = "#6B7280"
GRID_COLOR = "#D7DEE8"


def configure_style() -> None:
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#AAB4C3",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    sns.set_theme(style="whitegrid", rc={"font.family": plt.rcParams["font.family"]})


def paired_values(df: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray]:
    winners = df.loc[df["win"] == 1, ["game_id", feature]].set_index("game_id")
    losers = df.loc[df["win"] == 0, ["game_id", feature]].set_index("game_id")
    paired = winners.join(losers, lsuffix="_winner", rsuffix="_loser", how="inner").dropna()
    return paired[f"{feature}_winner"].to_numpy(), paired[f"{feature}_loser"].to_numpy()


def rank_biserial_paired(diff: np.ndarray) -> float:
    nonzero = np.asarray(diff, dtype=float)
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero != 0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def effect_magnitude(value: float, effect_type: str) -> str:
    value = abs(value)
    thresholds = (0.2, 0.5, 0.8) if effect_type == "Cohen's dz" else (0.1, 0.3, 0.5)
    if value < thresholds[0]:
        return "negligible"
    if value < thresholds[1]:
        return "small"
    if value < thresholds[2]:
        return "medium"
    return "large"


def bootstrap_rank_biserial_ci(
    diff: np.ndarray, rng: np.random.Generator, iterations: int = 1200
) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    values = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(diff, size=len(diff), replace=True)
        values[i] = rank_biserial_paired(sample)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("전체", df), ("승리", df[df["win"] == 1]), ("패배", df[df["win"] == 0])]
    for group_name, group in groups:
        for feature in ANALYSIS_FEATURES:
            series = group[feature].dropna()
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "n": len(series),
                    "mean": series.mean(),
                    "std": series.std(ddof=1),
                    "median": series.median(),
                    "q1": series.quantile(0.25),
                    "q3": series.quantile(0.75),
                    "min": series.min(),
                    "max": series.max(),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260804)
    rows: list[dict[str, object]] = []
    for feature in ANALYSIS_FEATURES:
        winner, loser = paired_values(df, feature)
        diff = winner - loser
        constant_difference = bool(np.allclose(diff, diff[0]))
        if constant_difference:
            shapiro_statistic, shapiro_pvalue = 1.0, 1.0
            normal = True
            test_name = "no_paired_variation"
            test_effect_type = "rank_biserial"
            test_effect = 0.0
            test = type("TestResult", (), {"statistic": 0.0, "pvalue": 1.0})()
        else:
            shapiro = stats.shapiro(diff)
            shapiro_statistic, shapiro_pvalue = float(shapiro.statistic), float(shapiro.pvalue)
            normal = bool(shapiro.pvalue >= 0.05)
        if not constant_difference and normal:
            test = stats.ttest_rel(winner, loser, nan_policy="omit")
            test_name = "paired_t_test"
            test_effect_type = "Cohen's dz"
            test_effect = cohen_dz(diff)
        elif not constant_difference:
            try:
                test = stats.wilcoxon(winner, loser, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            test_name = "wilcoxon_signed_rank"
            test_effect_type = "rank_biserial"
            test_effect = rank_biserial_paired(diff)
            test = type("TestResult", (), {"statistic": statistic, "pvalue": p_value})()

        rb = rank_biserial_paired(diff)
        ci_low, ci_high = bootstrap_rank_biserial_ci(diff, rng)
        rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "n_pairs": len(diff),
                "winner_mean": np.mean(winner),
                "loser_mean": np.mean(loser),
                "mean_difference_winner_minus_loser": np.mean(diff),
                "median_difference_winner_minus_loser": np.median(diff),
                "shapiro_w_difference": shapiro_statistic,
                "shapiro_p_difference": shapiro_pvalue,
                "normality_pass_0_05": normal,
                "selected_test": test_name,
                "test_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "cohens_dz": cohen_dz(diff),
                "rank_biserial": rb,
                "rank_biserial_ci_low": ci_low,
                "rank_biserial_ci_high": ci_high,
                "selected_effect_type": test_effect_type,
                "selected_effect_size": test_effect,
                "selected_effect_magnitude": effect_magnitude(test_effect, test_effect_type),
                "ranking_effect_magnitude": effect_magnitude(rb, "rank_biserial"),
            }
        )

    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = multipletests(result["p_value"], alpha=0.05, method="fdr_bh")[1]
    result["significant_raw_0_05"] = result["p_value"] < 0.05
    result["significant_fdr_0_05"] = result["p_fdr_bh"] < 0.05
    return result.sort_values("rank_biserial", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def compute_vif(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.dropna().astype(float)
    scaled = StandardScaler().fit_transform(clean)
    rows = []
    for i, feature in enumerate(clean.columns):
        value = variance_inflation_factor(scaled, i)
        rows.append({"feature": feature, "feature_label": FEATURE_LABELS[feature], "vif": float(value)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def iterative_vif_selection(df: pd.DataFrame, threshold: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = ANALYSIS_FEATURES.copy()
    history: list[dict[str, object]] = []
    initial = compute_vif(df[features])
    step = 0
    while len(features) > 4:
        current = compute_vif(df[features])
        worst = current.iloc[0]
        if np.isfinite(worst["vif"]) and worst["vif"] < threshold:
            break
        step += 1
        history.append(
            {
                "step": step,
                "removed_feature": worst["feature"],
                "removed_feature_label": worst["feature_label"],
                "vif_at_removal": worst["vif"],
                "remaining_after_removal": len(features) - 1,
            }
        )
        features.remove(str(worst["feature"]))
    final = compute_vif(df[features])
    return initial, pd.DataFrame(history), final


def run_pca(df: pd.DataFrame, features: list[str]) -> tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    clean = df[["game_id", "team", "season", "win", *features]].dropna().reset_index(drop=True)
    scaled = StandardScaler().fit_transform(clean[features])
    pca = PCA().fit(scaled)
    scores = pca.transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_80 = int(np.argmax(cumulative >= 0.80) + 1)
    pc_names = [f"PC{i + 1}" for i in range(len(features))]
    variance = pd.DataFrame(
        {
            "component": pc_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
            "retained_for_80pct": [i < n_80 for i in range(len(features))],
        }
    )
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=pc_names).reset_index(names="feature")
    loadings.insert(1, "feature_label", loadings["feature"].map(FEATURE_LABELS))
    score_frame = pd.concat(
        [clean[["game_id", "team", "season", "win"]], pd.DataFrame(scores, columns=pc_names)], axis=1
    )
    return pca, variance, loadings, score_frame, n_80


def save_distribution_plot(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(17, 18))
    for ax, feature in zip(axes.flat, ANALYSIS_FEATURES):
        winner = df.loc[df["win"] == 1, feature].dropna()
        loser = df.loc[df["win"] == 0, feature].dropna()
        values = pd.concat([winner, loser])
        bins = np.histogram_bin_edges(values, bins="fd")
        ax.hist(loser, bins=bins, density=True, alpha=0.50, color=LOSS_COLOR, label="패배")
        ax.hist(winner, bins=bins, density=True, alpha=0.50, color=WIN_COLOR, label="승리")
        ax.set_title(FEATURE_LABELS[feature])
        ax.set_ylabel("밀도")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    for ax in axes.flat[len(ANALYSIS_FEATURES) :]:
        ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=WIN_COLOR, alpha=0.65, label="승리팀"),
        plt.Rectangle((0, 0), 1, 1, color=LOSS_COLOR, alpha=0.65, label="패배팀"),
    ]
    fig.suptitle("승리팀과 패배팀의 경기 스탯 분포", fontsize=19, fontweight="bold", y=0.995)
    fig.text(0.5, 0.975, "한 행은 팀-경기 관측치이며 연장전 누적 스탯은 40분 기준으로 보정", ha="center", color=NEUTRAL)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_qq_plot(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(17, 18))
    for ax, feature in zip(axes.flat, ANALYSIS_FEATURES):
        winner, loser = paired_values(df, feature)
        stats.probplot(winner - loser, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(WIN_COLOR)
        ax.get_lines()[0].set_markeredgecolor("white")
        ax.get_lines()[0].set_markersize(3.5)
        ax.get_lines()[1].set_color(LOSS_COLOR)
        ax.set_title(FEATURE_LABELS[feature])
        ax.set_xlabel("이론적 분위수")
        ax.set_ylabel("관측 분위수")
        ax.grid(color=GRID_COLOR, linewidth=0.6)
    for ax in axes.flat[len(ANALYSIS_FEATURES) :]:
        ax.axis("off")
    fig.suptitle("경기별 승리팀-패배팀 차이의 Q-Q plot", fontsize=19, fontweight="bold", y=0.995)
    fig.text(0.5, 0.975, "점들이 직선에서 크게 벗어나면 대응표본 차이의 정규성이 약함", ha="center", color=NEUTRAL)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_boxplots(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(6, 3, figsize=(15, 22))
    for ax, feature in zip(axes.flat, ANALYSIS_FEATURES):
        loser = df.loc[df["win"] == 0, feature].dropna()
        winner = df.loc[df["win"] == 1, feature].dropna()
        bp = ax.boxplot([loser, winner], tick_labels=["패배", "승리"], patch_artist=True, widths=0.62, showfliers=True)
        for patch, color in zip(bp["boxes"], [LOSS_COLOR, WIN_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
        for median in bp["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.6)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=2.4, alpha=0.25, markeredgecolor=NEUTRAL)
        ax.set_title(FEATURE_LABELS[feature])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    fig.suptitle("승리팀 vs 패배팀 스탯 분포 비교", fontsize=19, fontweight="bold", y=0.995)
    fig.text(0.5, 0.975, "중앙선은 중앙값, 상자는 사분위 범위, 점은 이상치", ha="center", color=NEUTRAL)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_effect_ranking(tests: pd.DataFrame, output: Path) -> None:
    plot = tests.sort_values("rank_biserial").copy()
    colors = [LOSS_COLOR if value < 0 else WIN_COLOR for value in plot["rank_biserial"]]
    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(plot))
    errors = np.vstack(
        [
            plot["rank_biserial"] - plot["rank_biserial_ci_low"],
            plot["rank_biserial_ci_high"] - plot["rank_biserial"],
        ]
    )
    ax.barh(y, plot["rank_biserial"], color=colors, alpha=0.86)
    ax.errorbar(plot["rank_biserial"], y, xerr=errors, fmt="none", ecolor="#111827", capsize=2.5, linewidth=0.9)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.axvline(0.3, color=GRID_COLOR, linestyle="--", linewidth=1)
    ax.axvline(-0.3, color=GRID_COLOR, linestyle="--", linewidth=1)
    ax.set_yticks(y, plot["feature_label"])
    ax.set_xlabel("Matched-pairs rank-biserial correlation (95% bootstrap CI)")
    ax.set_title("승패 관련 스탯 효과크기 순위", fontsize=17, pad=15)
    ax.text(0.99, 0.02, "오른쪽: 승리팀이 높음  |  왼쪽: 승리팀이 낮음", transform=ax.transAxes, ha="right", color=NEUTRAL)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    for i, (_, row) in enumerate(plot.iterrows()):
        if row["significant_fdr_0_05"]:
            ax.text(0.985 if row["rank_biserial"] >= 0 else -0.985, i, "FDR<.05", va="center", ha="right" if row["rank_biserial"] >= 0 else "left", fontsize=7, color=NEUTRAL)
    ax.set_xlim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    corr = df[ANALYSIS_FEATURES].corr(method="spearman")
    labels = [FEATURE_LABELS[c] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.75},
        ax=ax,
    )
    ax.set_title("경기 스탯 상관관계", fontsize=18, pad=18)
    ax.tick_params(axis="x", rotation=55)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)
    corr.index.name = "feature"
    return corr


def save_vif_plot(initial: pd.DataFrame, final: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, frame, title in zip(axes, [initial, final], ["초기 변수", "VIF<10 반복 정리 후"]):
        plot = frame.sort_values("vif")
        finite = plot["vif"].replace([np.inf, -np.inf], np.nan)
        cap = max(12.0, float(finite.max(skipna=True) * 1.1))
        values = plot["vif"].replace(np.inf, cap)
        colors = [LOSS_COLOR if value >= 10 else WIN_COLOR for value in values]
        ax.barh(np.arange(len(plot)), values, color=colors, alpha=0.82)
        ax.set_yticks(np.arange(len(plot)), plot["feature_label"])
        ax.axvline(10, color="#111827", linestyle="--", linewidth=1.2, label="VIF=10")
        ax.set_xscale("log")
        ax.set_xlabel("VIF (로그 축)")
        ax.set_title(title)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("다중공선성 진단", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_pca_variance(variance: pd.DataFrame, n_80: int, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(1, len(variance) + 1)
    ax1.bar(x, variance["explained_variance_ratio"], color=WIN_COLOR, alpha=0.78)
    ax1.set_xlabel("주성분")
    ax1.set_ylabel("개별 설명력", color=WIN_COLOR)
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, variance["cumulative_explained_variance"], color=LOSS_COLOR, marker="o", linewidth=2)
    ax2.axhline(0.80, color="#111827", linestyle="--", linewidth=1)
    ax2.axvline(n_80, color=GRID_COLOR, linestyle=":", linewidth=1.4)
    ax2.set_ylim(0, 1.04)
    ax2.set_ylabel("누적 설명력", color=LOSS_COLOR)
    ax1.set_title(f"PCA 설명력: 누적 80%에 {n_80}개 주성분 필요", fontsize=17, pad=15)
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_pca_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, variance: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    for win, color, label in [(0, LOSS_COLOR, "패배팀"), (1, WIN_COLOR, "승리팀")]:
        group = scores[scores["win"] == win]
        ax.scatter(group["PC1"], group["PC2"], s=18, alpha=0.28, color=color, label=label, edgecolors="none")
    loading_plot = loadings.copy()
    loading_plot["importance_pc12"] = loading_plot["PC1"].abs() + loading_plot["PC2"].abs()
    loading_plot = loading_plot.nlargest(min(8, len(loading_plot)), "importance_pc12")
    scale_x = scores["PC1"].abs().quantile(0.92) * 0.86
    scale_y = scores["PC2"].abs().quantile(0.92) * 0.86
    arrow_points: list[tuple[float, float, str]] = []
    for _, row in loading_plot.iterrows():
        x, y = row["PC1"] * scale_x, row["PC2"] * scale_y
        ax.arrow(0, 0, x, y, color="#172B4D", alpha=0.72, width=0.007, head_width=0.10, length_includes_head=True)
        arrow_points.append((x, y, row["feature_label"]))
    for side in (-1, 1):
        side_points = sorted([point for point in arrow_points if (point[0] >= 0) == (side > 0)], key=lambda point: point[1])
        adjusted_y: list[float] = []
        for _, y, _ in side_points:
            label_y = y
            if adjusted_y and label_y - adjusted_y[-1] < 0.28:
                label_y = adjusted_y[-1] + 0.28
            adjusted_y.append(label_y)
        for (x, y, label), label_y in zip(side_points, adjusted_y):
            label_x = x + side * 0.18
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(label_x, label_y),
                fontsize=8.5,
                ha="left" if side > 0 else "right",
                va="center",
                color="#172B4D",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
                arrowprops={"arrowstyle": "-", "color": "#64748B", "linewidth": 0.6},
            )
    pc1 = variance.loc[0, "explained_variance_ratio"] * 100
    pc2 = variance.loc[1, "explained_variance_ratio"] * 100
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title("PCA 바이플롯: 경기 스타일과 승패 분리", fontsize=18, pad=15)
    ax.legend(frameon=False)
    ax.grid(color=GRID_COLOR, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_team_momentum(df: pd.DataFrame, window: int, value: str, title: str, ylabel: str, output: Path) -> None:
    ordered = df.sort_values(["team", "season", "game_date", "game_id"]).copy()
    rolling_col = f"rolling_{value}_{window}_descriptive"
    ordered[rolling_col] = ordered.groupby(["team", "season"], sort=False)[value].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    teams = sorted(ordered["team"].unique())
    seasons = sorted(ordered["season"].unique())
    season_colors = {seasons[0]: WIN_COLOR, seasons[1]: LOSS_COLOR}
    fig, axes = plt.subplots(5, 2, figsize=(15, 18), sharex=True, sharey=True)
    for ax, team in zip(axes.flat, teams):
        team_data = ordered[ordered["team"] == team]
        for season in seasons:
            season_data = team_data[team_data["season"] == season]
            ax.plot(
                season_data["season_game_number"],
                season_data[rolling_col],
                color=season_colors[season],
                linewidth=1.8,
                label=season,
            )
        ax.set_title(team)
        ax.grid(color=GRID_COLOR, linewidth=0.6)
        ax.set_xlim(1, 54)
    if value == "win":
        axes[0, 0].set_ylim(-0.03, 1.03)
    for ax in axes[-1, :]:
        ax.set_xlabel("시즌 경기 번호")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [plt.Line2D([0], [0], color=season_colors[s], linewidth=2, label=s) for s in seasons]
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.text(0.5, 0.975, f"시즌별 초기화, 현재 경기를 포함한 최근 {window}경기 단순 이동평균", ha="center", color=NEUTRAL)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=len(seasons), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=185)
    plt.close(fig)


def top_pca_loadings(loadings: pd.DataFrame, component: str, n: int = 4) -> str:
    selected = loadings.assign(_absolute=loadings[component].abs()).nlargest(n, "_absolute")
    return ", ".join(f"{row.feature_label}({row[component]:+.2f})" for _, row in selected.iterrows())


def write_report(
    output: Path,
    tests: pd.DataFrame,
    vif_history: pd.DataFrame,
    vif_final: pd.DataFrame,
    variance: pd.DataFrame,
    loadings: pd.DataFrame,
    n_80: int,
    rolling_window: int,
) -> None:
    top = tests.reindex(tests["rank_biserial"].abs().sort_values(ascending=False).index).head(7)
    significant_count = int(tests["significant_fdr_0_05"].sum())
    paired_t_count = int((tests["selected_test"] == "paired_t_test").sum())
    wilcoxon_count = int((tests["selected_test"] == "wilcoxon_signed_rank").sum())
    no_variation_count = int((tests["selected_test"] == "no_paired_variation").sum())
    lines = [
        "# KBL 승패 관련 통계 분석 보고서",
        "",
        "## 핵심 요약",
        "",
        f"- 분석 단위는 540경기의 승리팀–패배팀 대응쌍입니다. 팀 경기 행은 1,080개입니다.",
        f"- 대응차이 정규성을 만족한 {paired_t_count}개 변수는 대응표본 t-검정, 비정규적인 {wilcoxon_count}개 변수는 Wilcoxon 부호순위 검정을 사용했습니다.",
        f"- 두 팀 값이 경기마다 동일해 대응차이가 없는 변수는 {no_variation_count}개이며, p=1과 효과크기 0으로 기록했습니다.",
        f"- Benjamini-Hochberg FDR 5% 기준으로 {significant_count}개 변수가 승패 집단에서 유의한 차이를 보였습니다.",
        "- 효과크기 순위는 검정 종류가 달라도 비교할 수 있도록 모든 변수에 공통으로 계산한 matched-pairs rank-biserial correlation을 사용했습니다.",
        f"- VIF 반복 진단 후 {len(vif_final)}개 변수가 남았고, 이 변수들로 PCA를 수행했습니다.",
        f"- 누적 설명력 80%에 필요한 주성분은 {n_80}개입니다.",
        "",
        "## 효과크기가 큰 변수",
        "",
        "|순위|스탯|승리팀 평균|패배팀 평균|Rank-biserial|95% CI|FDR p-value|",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"|{rank}|{row['feature_label']}|{row['winner_mean']:.3f}|{row['loser_mean']:.3f}|"
            f"{row['rank_biserial']:+.3f}|[{row['rank_biserial_ci_low']:+.3f}, {row['rank_biserial_ci_high']:+.3f}]|{row['p_fdr_bh']:.3g}|"
        )
    removed = ", ".join(vif_history["removed_feature_label"].tolist()) if not vif_history.empty else "없음"
    lines.extend(
        [
            "",
            "## 다중공선성 및 PCA",
            "",
            f"- VIF 10 이상을 반복 진단했을 때 제거 후보가 된 변수: {removed}",
            "- VIF는 변수 간 중복을 보여주는 진단값입니다. 농구적 의미가 중요한 변수는 자동 삭제하지 말고 모델 교차검증으로 최종 선택해야 합니다.",
            f"- PC1 주요 적재량: {top_pca_loadings(loadings, 'PC1')}",
            f"- PC2 주요 적재량: {top_pca_loadings(loadings, 'PC2')}",
            "- 주성분의 의미는 적재량 부호와 크기를 함께 보고 해석해야 하며, 축의 부호 자체는 반대로 뒤집혀도 의미가 동일합니다.",
            "",
            "## 분석 해석 시 주의사항",
            "",
            "1. 이 분석은 경기 후 스탯과 승패의 연관성을 설명하며 인과관계를 증명하지 않습니다.",
            "2. 득점과 공격·수비 효율은 경기 결과와 구조적으로 가까우므로 예측 중요도로 과대해석하면 안 됩니다.",
            "3. 목표 경기의 실제 스탯은 사전 승패 예측 모델에 넣을 수 없습니다. 모델에는 이전 경기 이동평균만 사용해야 합니다.",
            "4. Shapiro-Wilk는 표본이 커질수록 작은 비정규성에도 민감하므로 Q-Q plot과 효과크기를 함께 확인해야 합니다.",
            f"5. 팀별 모멘텀 그래프는 시즌마다 초기화하고 현재 경기를 포함한 최근 {rolling_window}경기 이동평균입니다.",
            "",
            "## 그림",
            "",
            "- [스탯 분포](figures/01_stat_distributions.png)",
            "- [대응차이 Q-Q plot](figures/02_paired_difference_qq.png)",
            "- [승패 집단 박스플롯](figures/03_win_loss_boxplots.png)",
            "- [효과크기 순위](figures/04_effect_size_ranking.png)",
            "- [상관 히트맵](figures/05_correlation_heatmap.png)",
            "- [VIF 진단](figures/06_vif_diagnostics.png)",
            "- [PCA 설명력](figures/07_pca_explained_variance.png)",
            "- [PCA 바이플롯](figures/08_pca_biplot.png)",
            f"- [팀별 최근 {rolling_window}경기 승률](figures/09_team_rolling_win_rate.png)",
            f"- [팀별 최근 {rolling_window}경기 점수 마진](figures/10_team_rolling_score_margin.png)",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="KBL 승패 관련 통계 분석과 시각화")
    parser.add_argument("--rolling-window", type=int, default=5, help="팀별 모멘텀 이동평균 경기 수")
    args = parser.parse_args()
    if args.rolling_window < 2:
        raise ValueError("rolling-window는 2 이상이어야 합니다.")

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / "team_games.csv"
    analysis_dir = project_root / "analysis"
    results_dir = analysis_dir / "results"
    figures_dir = analysis_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    df = pd.read_csv(input_path, parse_dates=["game_date"])
    missing_columns = [c for c in ["game_id", "team", "season", "game_date", "win", *ANALYSIS_FEATURES] if c not in df]
    if missing_columns:
        raise KeyError(f"필수 열이 없습니다: {missing_columns}")
    if len(df) != 1080 or df["game_id"].nunique() != 540:
        raise ValueError("예상한 540경기/1,080 팀 행 구조가 아닙니다.")
    if not (df.groupby("game_id")["win"].agg(["sum", "count"]) == [1, 2]).all().all():
        raise ValueError("각 경기에는 승리팀 1개와 총 2개 팀 행이 필요합니다.")

    descriptive = descriptive_statistics(df)
    tests = paired_tests(df)
    vif_initial, vif_history, vif_final = iterative_vif_selection(df)
    pca_features = vif_final["feature"].tolist()
    _, variance, loadings, scores, n_80 = run_pca(df, pca_features)

    descriptive.to_csv(results_dir / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(results_dir / "paired_win_loss_tests.csv", index=False, encoding="utf-8-sig")
    vif_initial.to_csv(results_dir / "vif_initial.csv", index=False, encoding="utf-8-sig")
    vif_history.to_csv(results_dir / "vif_selection_history.csv", index=False, encoding="utf-8-sig")
    vif_final.to_csv(results_dir / "vif_final.csv", index=False, encoding="utf-8-sig")
    variance.to_csv(results_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(results_dir / "pca_loadings.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(results_dir / "pca_scores.csv", index=False, encoding="utf-8-sig")

    save_distribution_plot(df, figures_dir / "01_stat_distributions.png")
    save_qq_plot(df, figures_dir / "02_paired_difference_qq.png")
    save_boxplots(df, figures_dir / "03_win_loss_boxplots.png")
    save_effect_ranking(tests, figures_dir / "04_effect_size_ranking.png")
    corr = save_correlation_heatmap(df, figures_dir / "05_correlation_heatmap.png")
    corr.to_csv(results_dir / "spearman_correlation_matrix.csv", encoding="utf-8-sig")
    save_vif_plot(vif_initial, vif_final, figures_dir / "06_vif_diagnostics.png")
    save_pca_variance(variance, n_80, figures_dir / "07_pca_explained_variance.png")
    save_pca_biplot(scores, loadings, variance, figures_dir / "08_pca_biplot.png")
    save_team_momentum(
        df,
        args.rolling_window,
        "win",
        f"팀별 최근 {args.rolling_window}경기 승률 추이",
        "이동 승률",
        figures_dir / "09_team_rolling_win_rate.png",
    )
    save_team_momentum(
        df,
        args.rolling_window,
        "score_margin_per40",
        f"팀별 최근 {args.rolling_window}경기 점수 마진 추이",
        "평균 점수 마진(40분)",
        figures_dir / "10_team_rolling_score_margin.png",
    )
    write_report(
        analysis_dir / "statistical_analysis_report.md",
        tests,
        vif_history,
        vif_final,
        variance,
        loadings,
        n_80,
        args.rolling_window,
    )

    print(f"analysis_complete=true")
    print(f"games={df['game_id'].nunique()}")
    print(f"team_rows={len(df)}")
    print(f"fdr_significant={int(tests['significant_fdr_0_05'].sum())}/{len(tests)}")
    print(f"normal_differences={int(tests['normality_pass_0_05'].sum())}/{len(tests)}")
    print(f"vif_final_features={len(vif_final)}")
    print(f"pca_components_for_80pct={n_80}")


if __name__ == "__main__":
    main()
