from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseForecaster(ABC):
    """
    Abstract Base Class (Interface) for all forecasting models.
    
    This enforces a strict contract: every model MUST implement 'fit' and 'predict'.
    This allows the main experiment runner to treat ARIMA (Statistical), 
    TimeGPT (API-based), and Llama-3 (LLM) exactly the same way (Polymorphism).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (dict): A dictionary containing all hyperparameters 
                           (e.g., season_length, context_window, api_key).
        """
        self.config = config
        # We store the name for logging purposes (e.g., "AutoARIMA", "Llama-3")
        self.name = config.get('model_name', 'UnknownModel')

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Updates the model's internal state with historical data.
        
        - For Statistical models (ARIMA): This calculates coefficients.
        - For Foundation models (TimeGPT): This stores the context history.
        - For LLMs: This updates the prompt context window.
        
        Args:
            df_train (pd.DataFrame): Dataframe with columns ['ds', 'y'].
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generates the forecast for the next 'horizon' steps.
        
        Args:
            horizon (int): Number of steps to predict (e.g., 1 for next day).
            
        Returns:
            pd.DataFrame: A DataFrame containing columns ['ds', 'y_pred'].
                          'ds' must contain the future dates.
        """
        pass