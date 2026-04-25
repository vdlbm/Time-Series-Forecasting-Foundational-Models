"""
Statistical significance tests for time series forecasting model comparison.

Provides:
- Stationarity tests (ADF + KPSS) for exploratory analysis.
- Diebold-Mariano test for pairwise predictive accuracy comparison.
- Pairwise DM matrix with Bonferroni correction for multiple comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_report(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Runs ADF and KPSS tests and cross-references results.

    ADF H0: The series has a unit root (non-stationary).
    KPSS H0: The series is stationary.

    Cross-reference interpretation:
    - ADF rejects + KPSS does not reject → Stationary
    - ADF does not reject + KPSS rejects → Non-stationary
    - Both reject → Trend-stationary (difference needed)
    - Neither rejects → Inconclusive

    Args:
        series: Time series values (pd.Series or array-like).
        significance: Significance level for hypothesis testing.

    Returns:
        Dictionary with test statistics, p-values, and conclusion.
    """
    series = np.array(series).flatten()
    series = series[~np.isnan(series)]

    # ADF test
    adf_result = adfuller(series, autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_reject = adf_pvalue < significance

    # KPSS test (trend='c' for level stationarity)
    kpss_result = kpss(series, regression="c", nlags="auto")
    kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
    kpss_reject = kpss_pvalue < significance

    # Cross-reference
    if adf_reject and not kpss_reject:
        conclusion = "Stationary"
    elif not adf_reject and kpss_reject:
        conclusion = "Non-stationary (unit root)"
    elif adf_reject and kpss_reject:
        conclusion = "Trend-stationary (differencing may help)"
    else:
        conclusion = "Inconclusive"

    return {
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "adf_reject_h0": bool(adf_reject),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        "kpss_reject_h0": bool(kpss_reject),
        "conclusion": conclusion,
    }


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy (bilateral).

    H0: Both models have equal predictive accuracy.
    H1: The models have different predictive accuracy.

    For h > 1, applies Newey-West autocorrelation correction truncated
    at lag h-1 (Harvey, Leybourne, Newbold, 1997).

    Args:
        errors_1: Forecast errors from model 1 (array of length T).
        errors_2: Forecast errors from model 2 (array of length T).
        horizon: Forecast horizon (for autocorrelation correction).
        loss: Loss function type ('squared' or 'absolute').

    Returns:
        (dm_statistic, p_value) — two-sided test.
    """
    e1 = np.array(errors_1).flatten()
    e2 = np.array(errors_2).flatten()

    if len(e1) != len(e2):
        raise ValueError(
            f"Error arrays must have equal length. Got {len(e1)} and {len(e2)}."
        )

    T = len(e1)

    # Compute loss differential
    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: '{loss}'. Use 'squared' or 'absolute'.")

    d_mean = np.mean(d)

    # Newey-West variance estimator
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0.0
    max_lag = max(horizon - 1, 0)

    for lag in range(1, max_lag + 1):
        gamma_k = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        gamma_sum += 2 * gamma_k

    variance = (gamma_0 + gamma_sum) / T

    if variance <= 0:
        return 0.0, 1.0  # Cannot distinguish

    dm_stat = d_mean / np.sqrt(variance)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return float(dm_stat), float(p_value)


def pairwise_dm_matrix(
    errors_dict: Dict[str, np.ndarray],
    baseline: str = "naive",
    horizon: int = 1,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Computes Diebold-Mariano test for each model vs. baseline,
    with Bonferroni correction for multiple comparisons.

    Args:
        errors_dict: {model_name: error_array} for all models including baseline.
        baseline: Name of the baseline model (must be a key in errors_dict).
        horizon: Forecast horizon for autocorrelation correction.
        significance: Nominal significance level (before Bonferroni).

    Returns:
        DataFrame with columns [Model, DM_stat, p_value, p_value_bonf,
        significant_005, significant_001].
    """
    if baseline not in errors_dict:
        raise ValueError(f"Baseline '{baseline}' not found in errors_dict.")

    baseline_errors = errors_dict[baseline]
    other_models = [k for k in errors_dict if k != baseline]
    n_comparisons = len(other_models)

    records = []
    for model_name in other_models:
        dm_stat, p_val = diebold_mariano_test(
            baseline_errors, errors_dict[model_name], horizon=horizon
        )
        p_bonf = min(p_val * n_comparisons, 1.0)  # Bonferroni correction

        records.append(
            {
                "Model": model_name,
                "DM_stat": dm_stat,
                "p_value": p_val,
                "p_value_bonferroni": p_bonf,
                "significant_005": p_bonf < 0.05,
                "significant_001": p_bonf < 0.01,
            }
        )

    return pd.DataFrame(records)
