import pandas as pd
from typing import Dict, Any
from statsforecast import StatsForecast
# CORRECCIÓN: Usamos AutoARIMA y AutoETS (que engloba Holt, SES, etc.)
from statsforecast.models import AutoARIMA, AutoETS

from .base import BaseForecaster

class ClassicalWrapper(BaseForecaster):
    """
    Wrapper for 'StatsForecast' library models.
    Handles ARIMA using AutoARIMA.
    Handles SES, Holt, and Holt-Winters using AutoETS configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 1. Parse Configuration
        model_name = config.get('model_name', 'AutoARIMA')
        season_length = config.get('season_length', 1)
        
        # 2. Map String names to Actual Classes using AutoETS
        # AutoETS model syntax: Error-Trend-Seasonality (e.g., 'ANN')
        # A=Additive, M=Multiplicative, N=None, Z=Auto
        
        if model_name == 'AutoARIMA':
            selected_model = AutoARIMA(season_length=season_length)
        
        elif model_name == 'SimpleExpSmoothing':
            # SES = Error(Add), Trend(None), Season(None)
            selected_model = AutoETS(model='ANN', season_length=1)
            
        elif model_name == 'Holt':
            # Holt = Error(Add), Trend(Add), Season(None)
            selected_model = AutoETS(model='AAN', season_length=1)
            
        elif model_name == 'HoltWinters':
            # HoltWinters = Error(Add), Trend(Add), Season(Add)
            # We force seasonality here
            selected_model = AutoETS(model='AAA', season_length=season_length)
            
        else:
            raise ValueError(f"Classical model '{model_name}' not supported.")

        # 3. Initialize the StatsForecast Container
        self.sf = StatsForecast(
            models=[selected_model],
            freq=config.get('freq', 'D'), 
            n_jobs=1 
        )
        
        self.last_train_df = None

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