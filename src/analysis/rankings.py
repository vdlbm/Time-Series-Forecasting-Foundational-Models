"""
Non-parametric ranking and comparison of forecasting models.

Implements:
- Friedman test for overall ranking significance.
- Nemenyi Critical Difference diagram for pairwise comparisons.

These are the standard tools used in M4/M5 forecasting competitions
and academic benchmarks (Demšar, 2006).
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/notebook
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def friedman_ranking(
    results_df: pd.DataFrame,
    metric_col: str,
    model_col: str = "Model",
    dataset_col: str = "Dataset",
    lower_is_better: bool = True,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Friedman test + average ranking across datasets.

    H0: All models perform equally (same average rank).
    H1: At least one model is significantly different.

    Args:
        results_df: DataFrame with one row per (model, dataset) combination.
        metric_col: Column name of the metric to rank (e.g., 'RMSE').
        model_col: Column containing model identifiers.
        dataset_col: Column containing dataset identifiers.
        lower_is_better: If True, rank 1 = lowest metric value.

    Returns:
        (chi2_stat, p_value, rankings_df) where rankings_df has
        columns [Model, avg_rank] sorted by avg_rank ascending.
    """
    # Pivot: rows = datasets, columns = models, values = metric
    pivot = results_df.pivot(
        index=dataset_col, columns=model_col, values=metric_col
    )

    # Rank within each dataset (row)
    ranks = pivot.rank(axis=1, ascending=lower_is_better, method="average")

    # Average rank per model
    avg_ranks = ranks.mean(axis=0).sort_values()
    rankings_df = avg_ranks.reset_index()
    rankings_df.columns = [model_col, "avg_rank"]

    # Friedman test requires at least 3 models and 2 datasets
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        return 0.0, 1.0, rankings_df

    # scipy.stats.friedmanchisquare expects one array per model (column)
    model_arrays = [ranks[col].values for col in ranks.columns]
    chi2, p_value = stats.friedmanchisquare(*model_arrays)

    return float(chi2), float(p_value), rankings_df


def nemenyi_cd_diagram(
    rankings_df: pd.DataFrame,
    n_datasets: int,
    alpha: float = 0.05,
    model_col: str = "Model",
    rank_col: str = "avg_rank",
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Generates a Critical Difference (CD) diagram (Demšar, 2006).

    Models connected by a horizontal bar are NOT significantly different.
    The CD threshold is computed using the Nemenyi post-hoc test.

    Args:
        rankings_df: DataFrame with columns [Model, avg_rank] sorted by rank.
        n_datasets: Number of datasets used for ranking.
        alpha: Significance level (0.05 or 0.10).
        model_col: Column with model names.
        rank_col: Column with average ranks.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    models = rankings_df[model_col].values
    ranks = rankings_df[rank_col].values
    k = len(models)  # number of models
    N = n_datasets

    # Critical values for Nemenyi test (q_alpha for alpha=0.05)
    # Approximation using Studentized Range distribution
    # q_alpha = q_{alpha, k, inf} / sqrt(2)
    q_alpha_table = {
        0.05: {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
            12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
        },
        0.10: {
            2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
            7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978,
            12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159,
        },
    }

    q_val = q_alpha_table.get(alpha, q_alpha_table[0.05]).get(k, 3.0)
    cd = q_val * np.sqrt(k * (k + 1) / (6.0 * N))

    # --- Draw diagram ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_yaxis()

    # Draw axis line
    ax.axhline(y=0.5, color="black", linewidth=1.5, xmin=0, xmax=1)

    # Tick marks for ranks
    for r in range(1, k + 1):
        ax.plot([r, r], [0.3, 0.7], color="black", linewidth=1.5)
        ax.text(r, 0.1, str(r), ha="center", va="bottom", fontsize=10)

    # Place models
    left_models = []
    right_models = []
    mid = (1 + k) / 2.0

    for i, (model, rank) in enumerate(zip(models, ranks)):
        if rank <= mid:
            left_models.append((model, rank))
        else:
            right_models.append((model, rank))

    # Draw left models (rank labels on left)
    for i, (model, rank) in enumerate(left_models):
        y_pos = 1.2 + i * 0.7
        ax.plot([rank, rank], [0.7, y_pos], color="black", linewidth=1)
        ax.plot([rank, 0.3], [y_pos, y_pos], color="black", linewidth=1)
        ax.text(0.2, y_pos, f"{model} ({rank:.2f})", ha="right", va="center", fontsize=9)

    # Draw right models (rank labels on right)
    for i, (model, rank) in enumerate(right_models):
        y_pos = 1.2 + i * 0.7
        ax.plot([rank, rank], [0.7, y_pos], color="black", linewidth=1)
        ax.plot([rank, k + 0.7], [y_pos, y_pos], color="black", linewidth=1)
        ax.text(
            k + 0.8, y_pos, f"({rank:.2f}) {model}", ha="left", va="center", fontsize=9
        )

    # Draw cliques (groups of models not significantly different)
    # Find all maximal groups where max_rank - min_rank < CD
    sorted_indices = np.argsort(ranks)
    sorted_ranks = ranks[sorted_indices]

    cliques = []
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                # Check if this pair extends an existing clique
                merged = False
                for clique in cliques:
                    if i in clique and j not in clique:
                        if sorted_ranks[j] - sorted_ranks[min(clique)] < cd:
                            clique.add(j)
                            merged = True
                    elif j in clique and i not in clique:
                        if sorted_ranks[max(clique)] - sorted_ranks[i] < cd:
                            clique.add(i)
                            merged = True
                if not merged:
                    cliques.append({i, j})

    # Merge overlapping cliques
    merged_cliques = []
    for clique in cliques:
        merged = False
        for mc in merged_cliques:
            if clique & mc:
                mc.update(clique)
                merged = True
                break
        if not merged:
            merged_cliques.append(clique)

    # Draw clique bars
    bar_y = max(1.2 + len(left_models) * 0.7, 1.2 + len(right_models) * 0.7) + 0.3
    for clique in merged_cliques:
        if len(clique) < 2:
            continue
        clique_ranks = sorted_ranks[list(clique)]
        r_min, r_max = clique_ranks.min(), clique_ranks.max()
        ax.plot(
            [r_min, r_max], [bar_y, bar_y],
            color="red", linewidth=3, solid_capstyle="round",
        )
        bar_y += 0.35

    # CD annotation
    ax.annotate(
        f"CD = {cd:.2f}",
        xy=(mid, -0.3),
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_axis_off()
    ax.set_title(
        f"Critical Difference Diagram (α={alpha})", fontsize=13, fontweight="bold", pad=20
    )
    fig.tight_layout()
    return fig
