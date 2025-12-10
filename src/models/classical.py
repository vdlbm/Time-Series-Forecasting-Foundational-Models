import pandas as pd
from typing import Dict, Any
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SimpleExpSmoothing, Holt, HoltWinters

from .base import BaseForecaster

class ClassicalWrapper(BaseForecaster):
    """
    Wrapper for 'StatsForecast' library models.
    Handles ARIMA, Exponential Smoothing (SES), Holt, and Holt-Winters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 1. Parse Configuration
        model_name = config.get('model_name', 'AutoARIMA')
        season_length = config.get('season_length', 1)
        # Some models use 'alpha' (smoothing factor), defaulting to None lets the model optimize it
        alpha = config.get('alpha', None) 
        
        # 2. Map String names to Actual Classes
        # This allows us to instantiate the correct math model based on the YAML string
        if model_name == 'AutoARIMA':
            selected_model = AutoARIMA(season_length=season_length)
        
        elif model_name == 'SimpleExpSmoothing':
            selected_model = SimpleExpSmoothing(alpha=alpha)
            
        elif model_name == 'Holt':
            selected_model = Holt(alpha=alpha)
            
        elif model_name == 'HoltWinters':
            selected_model = HoltWinters(season_length=season_length)
            
        else:
            raise ValueError(f"Classical model '{model_name}' not supported.")

        # 3. Initialize the StatsForecast Container
        # We wrap the single selected model in the container to handle fitting logic efficiently
        self.sf = StatsForecast(
            models=[selected_model],
            freq=config.get('freq', 'D'), # Default to Daily if not specified
            n_jobs=1 # Sequential execution avoids conflicts inside the Rolling Window loop
        )
        
        self.last_train_df = None
        self.model_col_name = str(selected_model) # To find the column later (e.g., "AutoARIMA")

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Prepares the data for StatsForecast.
        The library requires a 'unique_id' column for panel data compatibility.
        """
        self.last_train_df = df_train.copy()
        # We add a dummy ID because it is required by StatsForecast even for univariate series
        self.last_train_df['unique_id'] = 'TargetSeries'

    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Fits and forecasts.
        """
        if self.last_train_df is None:
            raise ValueError("Model must be fit before predicting.")

        # 1. Fit & Predict using StatsForecast
        # We call fit inside predict because classical models are fast 
        # and need re-calculation for every new window in the rolling loop.
        self.sf.fit(self.last_train_df)
        preds = self.sf.predict(h=horizon)
        
        # 2. Format Output
        # StatsForecast returns columns like ['ds', 'AutoARIMA', 'unique_id']
        # We need to rename the model-specific column to generic 'y_pred'
        
        # Dynamic column finding: StatsForecast usually names the column as the model class repr
        # But sometimes it varies. We look for the column that is NOT 'ds' or 'unique_id'
        cols = [c for c in preds.columns if c not in ['ds', 'unique_id']]
        target_col = cols[0]
        
        out_df = preds[['ds', target_col]].rename(columns={target_col: 'y_pred'})
        
        return out_df