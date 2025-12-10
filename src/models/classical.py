import pandas as pd
from typing import Dict, Any
from statsforecast import StatsForecast
# UPDATED IMPORTS: We removed SimpleExpSmoothing and Holt, and added ETS
from statsforecast.models import AutoARIMA, HoltWinters, ETS

from .base import BaseForecaster

class ClassicalWrapper(BaseForecaster):
    """
    Wrapper for 'StatsForecast' library models.
    Handles ARIMA, and uses ETS to simulate SES, Holt, and Holt-Winters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 1. Parse Configuration
        model_name = config.get('model_name', 'AutoARIMA')
        season_length = config.get('season_length', 1)
        
        # 2. Map String names to Actual Classes
        # In newer statsforecast versions, we use ETS (Error, Trend, Seasonality) 
        # to implement SES and Holt.
        
        if model_name == 'AutoARIMA':
            selected_model = AutoARIMA(season_length=season_length)
        
        elif model_name == 'SimpleExpSmoothing':
            # ETS with 'ANN' = Additive Error, No Trend, No Seasonality (Equivalent to SES)
            selected_model = ETS(model='ANN', season_length=1)
            
        elif model_name == 'Holt':
            # ETS with 'AAN' = Additive Error, Additive Trend, No Seasonality (Equivalent to Holt)
            selected_model = ETS(model='AAN', season_length=1)
            
        elif model_name == 'HoltWinters':
            # We can use the specific HoltWinters class or ETS('AAA')
            # Using the specific class as it's still available and robust
            selected_model = HoltWinters(season_length=season_length)
            
        else:
            raise ValueError(f"Classical model '{model_name}' not supported.")

        # 3. Initialize the StatsForecast Container
        self.sf = StatsForecast(
            models=[selected_model],
            freq=config.get('freq', 'D'), 
            n_jobs=1 
        )
        
        self.last_train_df = None
        self.model_col_name = str(selected_model) 

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Prepares the data for StatsForecast.
        """
        self.last_train_df = df_train.copy()
        # Dummy ID for single-series forecast
        self.last_train_df['unique_id'] = 'TargetSeries'

    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Fits and forecasts.
        """
        if self.last_train_df is None:
            raise ValueError("Model must be fit before predicting.")

        # 1. Fit & Predict
        self.sf.fit(self.last_train_df)
        preds = self.sf.predict(h=horizon)
        
        # 2. Format Output
        # Dynamic column finding (exclude ds and unique_id)
        cols = [c for c in preds.columns if c not in ['ds', 'unique_id']]
        target_col = cols[0]
        
        out_df = preds[['ds', target_col]].rename(columns={target_col: 'y_pred'})
        
        return out_df