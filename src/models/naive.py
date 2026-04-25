import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """
    Random Walk (Naive) baseline forecaster.

    Predicts that future values will equal the last observed value.
    This is the canonical benchmark for financial time series:
    any model that cannot beat the naive forecast adds no value.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.last_value: float = 0.0
        self.last_date: pd.Timestamp = pd.NaT
        self.freq: str = config.get("freq", "D")

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Stores the last observed value and date from training data.
        """
        self.last_value = float(df_train["y"].iloc[-1])
        self.last_date = pd.to_datetime(df_train["ds"].iloc[-1])

    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generates a flat forecast repeating the last observed value.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            DataFrame with columns ['ds', 'y_pred'].
        """
        future_dates = pd.date_range(
            start=self.last_date, periods=horizon + 1, freq=self.freq
        )[1:]

        return pd.DataFrame(
            {"ds": future_dates, "y_pred": [self.last_value] * horizon}
        )
