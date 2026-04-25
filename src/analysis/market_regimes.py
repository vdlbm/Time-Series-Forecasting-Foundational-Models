"""
Market regime detection using Hidden Markov Models.

Identifies bull/bear (low-vol/high-vol) market regimes from price data
and segments forecasting metrics by regime. This answers the central
TFG question: "Do foundation models outperform classical ones during
volatile / bearish market conditions?"
"""

import numpy as np
import pandas as pd
from typing import List
from hmmlearn.hmm import GaussianHMM


def detect_regimes(
    prices: pd.Series, n_states: int = 2, random_seed: int = 42
) -> pd.Series:
    """
    Fits a 2-state Gaussian HMM on log-returns to detect market regimes.

    States are ordered by mean return:
    - State 0 = Bull / low-volatility (higher mean return)
    - State 1 = Bear / high-volatility (lower mean return)

    Args:
        prices: Price series with datetime index.
        n_states: Number of HMM states (default 2: bull/bear).
        random_seed: For reproducibility.

    Returns:
        pd.Series of regime labels (0=bull, 1=bear), indexed by date.
        The first date is dropped (no return for the first observation).
    """
    prices = prices.dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    X = log_returns.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_seed,
    )
    model.fit(X)
    states = model.predict(X)

    # Order states: state with highest mean return → 0 (bull)
    means = model.means_.flatten()
    state_order = np.argsort(-means)  # descending by mean return
    state_map = {old: new for new, old in enumerate(state_order)}
    ordered_states = np.array([state_map[s] for s in states])

    return pd.Series(ordered_states, index=log_returns.index, name="regime")


def metrics_by_regime(
    detailed_df: pd.DataFrame,
    regimes: pd.Series,
    metric_cols: List[str],
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Segments forecasting metrics by market regime.

    Merges the regime labels with the detailed predictions DataFrame
    and computes the mean of each metric grouped by regime.

    Args:
        detailed_df: Per-fold prediction results with a date column and metrics.
        regimes: pd.Series of regime labels (0=bull, 1=bear) with datetime index.
        metric_cols: List of metric column names to aggregate (e.g., ['RMSE', 'MAPE']).
        date_col: Name of the date column in detailed_df.

    Returns:
        DataFrame with columns [Model, Regime, <metric_cols>...].
    """
    df = detailed_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Map each prediction date to the nearest regime date
    regime_df = regimes.reset_index()
    regime_df.columns = ["regime_date", "regime"]

    # Merge using asof: find the closest regime date <= prediction date
    df = df.sort_values(date_col)
    regime_df = regime_df.sort_values("regime_date")

    merged = pd.merge_asof(
        df, regime_df, left_on=date_col, right_on="regime_date", direction="backward"
    )

    # Label regimes for readability
    merged["Regime"] = merged["regime"].map({0: "Bull", 1: "Bear"})

    # Aggregate
    agg_cols = ["Model", "Regime"]
    result = merged.groupby(agg_cols)[metric_cols].mean().reset_index()

    return result
