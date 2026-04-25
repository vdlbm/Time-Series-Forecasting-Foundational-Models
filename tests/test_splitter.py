import pytest
import pandas as pd
import numpy as np
from src.evaluation.splitter import ExpandingWindowSplitter


class TestExpandingWindowSplitter:

    def test_basic_split_monthly(self):
        """Monthly data from 2022: 1 year train, monthly steps, h=1."""
        dates = pd.date_range(start="2022-01-01", periods=48, freq="MS")
        df = pd.DataFrame({"ds": dates, "y": range(48)})

        splitter = ExpandingWindowSplitter(
            initial_train_end="2022-12-31", fold_step="1M", test_horizon=1
        )
        splits = list(splitter.split(df))

        # Train should grow with each fold
        assert len(splits) > 0
        assert len(splits[0][0]) == 12  # Jan-Dec 2022
        assert len(splits[0][1]) == 1   # Jan 2023
        assert len(splits[-1][0]) > len(splits[0][0])  # Expanding

    def test_expanding_train_grows(self):
        """Verify that each successive fold has a larger training set."""
        dates = pd.date_range(start="2022-01-01", periods=48, freq="MS")
        df = pd.DataFrame({"ds": dates, "y": range(48)})

        splitter = ExpandingWindowSplitter(
            initial_train_end="2022-12-31", fold_step="1M", test_horizon=1
        )
        prev_size = 0
        for train, test in splitter.split(df):
            assert len(train) >= prev_size
            prev_size = len(train)

    def test_no_data_leakage(self, sample_data):
        """Train max date must be strictly before test min date."""
        splitter = ExpandingWindowSplitter(
            initial_train_end="2023-01-15", fold_step="1W", test_horizon=1
        )
        for train, test in splitter.split(sample_data):
            assert train["ds"].max() < test["ds"].min()

    def test_horizon_respected(self):
        """Each test set must have exactly test_horizon rows."""
        dates = pd.date_range(start="2022-01-01", periods=48, freq="MS")
        df = pd.DataFrame({"ds": dates, "y": range(48)})

        for h in [1, 3]:
            splitter = ExpandingWindowSplitter(
                initial_train_end="2022-12-31", fold_step="1M", test_horizon=h
            )
            for train, test in splitter.split(df):
                assert len(test) == h

    def test_weekly_step_with_business_days(self, sample_data_business_days):
        """Weekly step on business day data (simulates SP500)."""
        splitter = ExpandingWindowSplitter(
            initial_train_end="2022-06-30", fold_step="1W", test_horizon=5
        )
        splits = list(splitter.split(sample_data_business_days))
        assert len(splits) > 0

        # Each test should have 5 business days
        for train, test in splits:
            assert len(test) == 5

    def test_handles_gaps_holidays(self, sample_data_with_gaps):
        """Splitter handles missing dates (holidays) gracefully."""
        splitter = ExpandingWindowSplitter(
            initial_train_end="2023-01-31", fold_step="1W", test_horizon=1
        )
        splits = list(splitter.split(sample_data_with_gaps))
        assert len(splits) > 0

        for train, test in splits:
            assert len(test) == 1
            # Test date should be the first available after boundary
            assert test["ds"].iloc[0] > train["ds"].iloc[-1]

    def test_count_folds_matches_split(self):
        """count_folds() must return the same number as len(list(split()))."""
        dates = pd.date_range(start="2022-01-01", periods=48, freq="MS")
        df = pd.DataFrame({"ds": dates, "y": range(48)})

        splitter = ExpandingWindowSplitter(
            initial_train_end="2022-12-31", fold_step="1M", test_horizon=1
        )
        assert splitter.count_folds(df) == len(list(splitter.split(df)))

    def test_insufficient_data_yields_zero_folds(self):
        """If all data is before initial_train_end, no folds are produced."""
        dates = pd.date_range(start="2022-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"ds": dates, "y": range(6)})

        splitter = ExpandingWindowSplitter(
            initial_train_end="2025-12-31", fold_step="1M", test_horizon=1
        )
        assert splitter.count_folds(df) == 0
        assert len(list(splitter.split(df))) == 0
