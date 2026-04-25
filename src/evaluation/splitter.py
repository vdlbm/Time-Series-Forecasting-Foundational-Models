import pandas as pd
from typing import Generator, Tuple


class ExpandingWindowSplitter:
    """
    Date-based Expanding Window splitter for time series backtesting.

    Instead of splitting by integer index, splits are driven by calendar dates.
    This avoids issues with leap years, market holidays, and irregular frequencies.

    Strategy:
    - Training window GROWS with each fold (expanding window).
    - The boundary advances by a fixed calendar offset (e.g., 1 week, 1 month).
    - Missing dates (holidays, weekends) are handled implicitly: the train set
      includes all data points up to the boundary, and the test set starts at
      the first available data point after the boundary.
    """

    def __init__(self, initial_train_end: str, fold_step: str, test_horizon: int):
        """
        Args:
            initial_train_end: Date string (e.g., '2022-12-31'). The first fold
                               trains on all data up to this date.
            fold_step: Calendar offset string. '1W' for 1 week, '1M' for 1 month.
            test_horizon: Number of data points to include in the test set.
        """
        self.initial_train_end = pd.Timestamp(initial_train_end)
        self.test_horizon = test_horizon

        # Parse fold_step into a pd.DateOffset
        if fold_step.upper() == "1W":
            self.fold_offset = pd.DateOffset(weeks=1)
        elif fold_step.upper() == "1M":
            self.fold_offset = pd.DateOffset(months=1)
        else:
            raise ValueError(
                f"Unsupported fold_step: '{fold_step}'. Use '1W' or '1M'."
            )

    def count_folds(self, df: pd.DataFrame) -> int:
        """
        Counts the total number of folds without generating data.

        Args:
            df: DataFrame with a 'ds' datetime column.

        Returns:
            Number of folds that can be generated.
        """
        n = 0
        fold_idx = 0
        while True:
            boundary = self.initial_train_end + self.fold_offset * fold_idx
            candidates = df[df["ds"] > boundary]
            if len(candidates) < self.test_horizon:
                break
            # Also verify there is at least 1 training point
            train = df[df["ds"] <= boundary]
            if len(train) < 1:
                fold_idx += 1
                continue
            n += 1
            fold_idx += 1
        return n

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yields (train, test) tuples with expanding training windows.

        Args:
            df: DataFrame with columns ['ds', 'y']. 'ds' must be datetime.
                The DataFrame should be sorted by 'ds' ascending.

        Yields:
            (train_df, test_df) where:
            - train_df: all rows with ds <= boundary (grows each fold)
            - test_df: the next `test_horizon` rows after boundary
        """
        if "ds" not in df.columns:
            raise ValueError("DataFrame must contain a 'ds' column.")

        # Ensure sorted
        df = df.sort_values("ds").reset_index(drop=True)

        fold_idx = 0
        while True:
            boundary = self.initial_train_end + self.fold_offset * fold_idx

            # Train: all data up to and including boundary
            train = df[df["ds"] <= boundary].copy()

            # Test: first `test_horizon` points strictly after boundary
            candidates = df[df["ds"] > boundary]

            if len(candidates) < self.test_horizon:
                break  # Not enough future data for this fold

            if len(train) < 1:
                fold_idx += 1
                continue  # No training data yet, skip

            test = candidates.iloc[: self.test_horizon].copy()

            yield train, test
            fold_idx += 1
