import pandas as pd
from typing import Optional

class TimeSeriesLoader:
    """
    Handles ingestion and preprocessing of time series data.
    Standardizes formats to 'ds' (datetime) and 'y' (target) for downstream compatibility.
    """
    
    def __init__(self, file_path: str, time_col: str = 'DateTime', target_col: str = 'Open'):
        
        # Lazy loading of data
        self.file_path = file_path
        self.time_col = time_col
        self.target_col = target_col
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> 'TimeSeriesLoader':
        """Loads CSV, standardizes columns, and sets datetime index."""
        try:
            self.df = pd.read_csv(self.file_path)
            
            # Defensive programming: Check if columns exist
            if self.time_col not in self.df.columns or self.target_col not in self.df.columns:
                available = self.df.columns.tolist()
                raise ValueError(f"Columns {self.time_col}/{self.target_col} not found. Available: {available}")

            # Rename columns to standard 'ds' and 'y'
            rename_map = {self.time_col: 'ds', self.target_col: 'y'}
            self.df = self.df.rename(columns=rename_map)
            
            # Convert 'ds' to datetime, set as index and order the TS in case it's unordered
            self.df['ds'] = pd.to_datetime(self.df['ds'])
            self.df = self.df.set_index('ds').sort_index()
            
            # Ensure 'y' is numeric
            self.df['y'] = pd.to_numeric(self.df['y'], errors='coerce') # coerce errors to NaN
            
            initial_rows = len(self.df)
            nan_rows = self.df['y'].isna().sum()
            
            if nan_rows > 0:
                print(f"WARNING: Found {nan_rows} rows with NaN values ({(nan_rows/initial_rows):.2%}). Dropping them.")
                self.df = self.df.dropna(subset=['y'])
                print(f"Data cleaned. Rows remaining: {len(self.df)}")
            else:
                print(f"Data loaded successfully. No NaNs found. Shape: {self.df.shape}")

            return self
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

    def resample(self, freq: str, agg_method: str = 'last') -> pd.DataFrame:
        """
        Resamples data to the desired frequency, handling missing values via forward fill.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        df_res = self.df.copy()
        resampler = df_res['y'].resample(freq)

        if agg_method == 'last':
            df_res = resampler.last()
        elif agg_method == 'mean':
            df_res = resampler.mean()
        elif agg_method == 'sum':
            df_res = resampler.sum()
        elif agg_method == 'first':
            df_res = resampler.first()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")

        # Forward fill is standard for financial time series to maintain state
        df_res = df_res.asfreq(freq, method='ffill')
        
        return df_res.reset_index()

    def split_by_date(self, df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simple split for initial validation checks. Classic train-test split by date."""
        mask = df['ds'] <= cutoff_date
        return df[mask].copy(), df[~mask].copy()