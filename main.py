"""
TFG: Comparative Analysis of Foundational Models and LLMs
for Zero-Shot Time Series Forecasting in Financial Decision-Making.

Experiment runner with:
- Date-based expanding window backtesting.
- Multi-horizon evaluation (h=1, h=3/5/7).
- 11 models: Naive + 4 Classical + 4 Foundation + 2 LLM.
- Financial metrics (Sharpe, MaxDD, Calmar) computed per model run.
- Checkpoint saving after each model to allow resumption.
"""

import yaml
import pandas as pd
import numpy as np
import time
import torch
import gc
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# Silence Nixtla logs
logging.getLogger("nixtla").setLevel(logging.ERROR)

from src.data.loader import TimeSeriesLoader
from src.models.factory import ModelFactory
from src.evaluation.splitter import ExpandingWindowSplitter
from src.evaluation.metrics import PerformanceEvaluator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def setup_logging() -> logging.Logger:
    """Configure logging to both console and timestamped file."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/experiment_{timestamp}.log"

    logger = logging.getLogger("tfg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_path}")
    return logger


def load_config(path: str = "config/experiments.yaml") -> dict:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_filter_dataset(
    ds_name: str, ds_config: dict, global_cfg: dict, logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    Load CSV, filter to [start_date, end_date), clean NaN.

    Returns DataFrame with columns ['ds', 'y'] (no index), sorted by ds.
    Returns None if loading fails.
    """
    try:
        loader = TimeSeriesLoader(
            file_path=ds_config["path"],
            time_col=global_cfg["time_col"],
            target_col=global_cfg["target_col"],
        )
        loaded = loader.load()
        df = loaded.df if hasattr(loaded, "df") else loaded

        # Ensure 'ds' is a column, not the index
        if "ds" not in df.columns:
            df = df.reset_index()
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

        # Filter to experiment window
        start = pd.Timestamp(global_cfg["start_date"])
        end = pd.Timestamp(global_cfg["end_date"])
        df = df[(df["ds"] >= start) & (df["ds"] < end)].copy()

        # Drop NaN
        df = df.dropna(subset=["y"])
        df = df.sort_values("ds").reset_index(drop=True)

        # Keep only ds and y
        df = df[["ds", "y"]].copy()

        logger.info(
            f"  Dataset loaded: {len(df)} rows "
            f"[{df['ds'].iloc[0].date()} → {df['ds'].iloc[-1].date()}]"
        )
        return df

    except Exception as e:
        logger.error(f"  FAILED to load dataset {ds_name}: {e}")
        return None


def build_splitter(
    global_cfg: dict, ds_config: dict, horizon: int
) -> ExpandingWindowSplitter:
    """Construct splitter from config."""
    return ExpandingWindowSplitter(
        initial_train_end=global_cfg["initial_train_end"],
        fold_step=ds_config["fold_step"],
        test_horizon=horizon,
    )


def prepare_train_for_model(
    train_df: pd.DataFrame,
    model_key: str,
    model_config: dict,
    freq: str,
) -> pd.DataFrame:
    """
    Prepare train DataFrame for a specific model.

    Handles TimeGPT unique_id, frequency assignment, and NaN cleaning.
    Does NOT modify the original DataFrame.
    """
    df = train_df.copy()

    # Clean residual NaN (forward-fill then back-fill)
    df["y"] = df["y"].ffill().bfill()
    if df["y"].isna().any():
        df["y"] = df["y"].fillna(0)

    if model_key.lower() == "timegpt":
        df["unique_id"] = "series_1"
        df = df[["unique_id", "ds", "y"]]

    return df


def run_model_on_folds(
    model,
    model_key: str,
    model_config: dict,
    splitter: ExpandingWindowSplitter,
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    logger: logging.Logger,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Execute all folds for one model on one dataset/horizon.

    Returns:
        (detailed_records, fold_metrics) where each is a list of dicts.
    """
    detailed_records: List[Dict] = []
    fold_metrics: List[Dict] = []

    for i, (train_df, test_df) in enumerate(splitter.split(df)):
        start_time = time.time()

        # Prepare train data
        prepared_train = prepare_train_for_model(
            train_df, model_key, model_config, freq
        )

        if len(prepared_train) < 2:
            continue

        # TimeGPT rate limit
        if model_key.lower() == "timegpt":
            time.sleep(1.5)

        # Retry logic
        max_retries = 3
        forecast_df = None
        last_error = None

        for attempt in range(max_retries):
            try:
                model.fit(prepared_train)
                forecast_df = model.predict(horizon=len(test_df))
                break
            except Exception as e:
                last_error = e
                if "429" in str(e):
                    time.sleep(5 * (attempt + 1))
                else:
                    time.sleep(1)

        if forecast_df is None:
            logger.warning(f"    Fold {i}: FAILED ({last_error})")
            continue

        inference_time = time.time() - start_time

        y_true = test_df["y"].values
        y_pred = forecast_df["y_pred"].values
        last_known_y = train_df["y"].iloc[-1]

        metrics = PerformanceEvaluator.calculate_metrics(
            y_true=y_true, y_pred=y_pred, previous_y=last_known_y
        )

        # Record details (first test date as reference)
        test_date = test_df["ds"].iloc[0]

        detailed_records.append(
            {
                "Date": test_date,
                "y_true": float(y_true[0]),
                "y_pred": float(y_pred[0]),
                "fold": i,
                "train_size": len(train_df),
                **metrics,
            }
        )

        metrics["inference_time"] = inference_time
        fold_metrics.append(metrics)

    return detailed_records, fold_metrics


def compute_financial_metrics(
    fold_metrics: List[Dict], periods_per_year: int
) -> Dict[str, float]:
    """Compute Sharpe, MaxDD, Calmar from per-fold strategy returns."""
    if not fold_metrics:
        return {"Sharpe_Ratio": 0.0, "Max_Drawdown": 0.0, "Calmar_Ratio": 0.0}

    returns = np.array([m["Strategy_Return_Pct"] for m in fold_metrics])

    return {
        "Sharpe_Ratio": PerformanceEvaluator.sharpe_ratio(returns, periods_per_year),
        "Max_Drawdown": PerformanceEvaluator.max_drawdown(returns),
        "Calmar_Ratio": PerformanceEvaluator.calmar_ratio(returns, periods_per_year),
    }


def save_checkpoint(records: List[Dict], path: str) -> None:
    """Save partial results to CSV."""
    if records:
        pd.DataFrame(records).to_csv(path, index=False)


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TFG EXPERIMENT: Expanding Window, Date-Based, Multi-Horizon")
    logger.info("=" * 60)

    cfg = load_config()
    global_cfg = cfg["global"]
    datasets_cfg = cfg["datasets"]
    models_cfg = cfg["models"]

    os.makedirs("results", exist_ok=True)

    summary_log: List[Dict] = []
    detailed_log: List[Dict] = []

    # ---- MAIN LOOP: Dataset → Horizon → Model ----
    for ds_name, ds_config in datasets_cfg.items():
        logger.info(f"\n{'█' * 60}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'█' * 60}")

        df = load_and_filter_dataset(ds_name, ds_config, global_cfg, logger)
        if df is None:
            continue

        freq = ds_config["frequency"]
        horizons = ds_config["horizons"]
        periods_per_year = ds_config["periods_per_year"]

        for horizon in horizons:
            splitter = build_splitter(global_cfg, ds_config, horizon)
            n_folds = splitter.count_folds(df)
            logger.info(f"\n  Horizon h={horizon} | Folds: {n_folds}")

            if n_folds < 1:
                logger.warning(f"  SKIPPED: Not enough data for h={horizon}")
                continue

            for model_key, model_config in models_cfg.items():
                logger.info(
                    f"    Model: {model_key.upper():<20} ⏳ Processing...",
                )

                local_config = model_config.copy()
                local_config["freq"] = freq
                local_config["frequency"] = freq

                if local_config["type"] == "classical":
                    local_config["season_length"] = ds_config["season_length"]
                if local_config["type"] == "llm_local":
                    local_config["dataset_description"] = ds_config.get(
                        "description", "Financial time series"
                    )

                try:
                    model = ModelFactory.get_model(local_config)

                    detailed_records, fold_metrics = run_model_on_folds(
                        model=model,
                        model_key=model_key,
                        model_config=local_config,
                        splitter=splitter,
                        df=df,
                        horizon=horizon,
                        freq=freq,
                        logger=logger,
                    )

                    if fold_metrics:
                        # Average technical metrics
                        avg = pd.DataFrame(fold_metrics).mean().to_dict()

                        # Financial metrics (over full returns series)
                        fin = compute_financial_metrics(fold_metrics, periods_per_year)

                        summary_entry = {
                            "Dataset": ds_name,
                            "Horizon": horizon,
                            "Model": model_key,
                            "Type": local_config["type"],
                            "n_folds": len(fold_metrics),
                            **avg,
                            **fin,
                        }
                        summary_log.append(summary_entry)

                        # Attach metadata to detailed records
                        for rec in detailed_records:
                            rec["Dataset"] = ds_name
                            rec["Horizon"] = horizon
                            rec["Model"] = model_key
                            rec["Type"] = local_config["type"]
                        detailed_log.extend(detailed_records)

                        logger.info(
                            f"    Model: {model_key.upper():<20} ✅ "
                            f"RMSE={avg['RMSE']:.4f} | "
                            f"MAPE={avg['MAPE']:.2f}% | "
                            f"DA={avg['Directional_Accuracy']:.1f}% | "
                            f"Sharpe={fin['Sharpe_Ratio']:.2f}"
                        )
                    else:
                        logger.warning(
                            f"    Model: {model_key.upper():<20} ⚠️ All folds failed"
                        )

                    # Cleanup
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Checkpoint after each model
                    save_checkpoint(
                        summary_log, f"results/summary_checkpoint.csv"
                    )

                except Exception as e:
                    logger.error(
                        f"    Model: {model_key.upper():<20} ❌ {str(e)[:120]}"
                    )
                    continue

    # ---- SAVE FINAL RESULTS ----
    logger.info("\n" + "=" * 60)
    logger.info("SAVING FINAL RESULTS")
    logger.info("=" * 60)

    if summary_log:
        summary_df = pd.DataFrame(summary_log)
        summary_df.to_csv("results/summary_metrics.csv", index=False)
        logger.info("→ results/summary_metrics.csv saved.")

        # Also save per-horizon summaries
        for h in summary_df["Horizon"].unique():
            h_df = summary_df[summary_df["Horizon"] == h]
            h_df.to_csv(f"results/summary_metrics_h{h}.csv", index=False)
            logger.info(f"→ results/summary_metrics_h{h}.csv saved.")

    if detailed_log:
        detailed_df = pd.DataFrame(detailed_log)
        detailed_df.to_csv("results/detailed_predictions.csv", index=False)
        logger.info("→ results/detailed_predictions.csv saved.")

    # Cleanup checkpoint
    checkpoint_path = "results/summary_checkpoint.csv"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
