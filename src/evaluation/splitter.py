import pandas as pd
from typing import Generator, Tuple

class RollingWindowSplitter:
    """
    Generates time-ordered train/test splits for Backtesting.
    
    Implements an 'Expanding Window' strategy:
    - Train set: Grows over time (includes all history up to cutoff).
    - Test set: Slides forward by 'step_size' (usually 1).
    """
    
    def __init__(self, n_windows: int, test_horizon: int, input_window_size: int = 365):
        """
        Args:
            n_windows (int): Total number of evaluation steps (e.g., 365 days).
            test_horizon (int): How many steps to predict in each window (e.g., 1 day).
            input_window_size (int): Minimum required history to start training.
        """
        self.n_windows = n_windows
        self.test_horizon = test_horizon
        self.min_train_size = input_window_size

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yields (train, test) tuples moving forward in time.
        """
        total_rows = len(df)
        
        # Calculate total samples needed
        # We need space for the Train history + the Test periods
        required_samples = self.min_train_size + (self.n_windows * self.test_horizon)
        
        if total_rows < required_samples:
            raise ValueError(
                f"Dataset too small. Needed {required_samples} rows, "
                f"but got {total_rows}. Reduce n_windows or input_window_size."
            )

        # We determine the starting point so that the LAST window ends exactly at the end of the dataframe.
        # This ensures we use the most recent data available.
        
        # Example: 1000 rows. n_windows=30. horizon=1.
        # Last test ends at 1000.
        # First test starts at 1000 - (30 * 1) = 970.
        
        start_test_idx = total_rows - (self.n_windows * self.test_horizon)
        
        for i in range(self.n_windows):
            # Calculate dynamic cutoff
            cutoff = start_test_idx + (i * self.test_horizon)
            end_test = cutoff + self.test_horizon
            
            # 1. Train Set: From beginning (or fixed window) up to cutoff
            # We use Expanding Window (df.iloc[:cutoff]) so ARIMA gets full history.
            # (Note: Llama-3 wrapper handles truncating this history internally to 90 days)
            train = df.iloc[:cutoff].copy()
            
            # 2. Test Set: The specific future horizon we want to predict
            test = df.iloc[cutoff:end_test].copy()
            
            yield train, test