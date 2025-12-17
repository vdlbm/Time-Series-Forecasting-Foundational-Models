import numpy as np
import pandas as pd
from typing import Dict, Union

class PerformanceEvaluator:
    """
    Computes Technical and Business metrics for Time Series Forecasting.
    Technical Metrics:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - MAPE (Mean Absolute Percentage Error)
    Business Metrics:
        - Directional Accuracy (DirAcc)
        
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, previous_y: float = None) -> Dict[str, float]:
        """
        Calculates a comprehensive suite of metrics.
        
        Args:
            y_true (np.ndarray): Actual values.
            y_pred (np.ndarray): Predicted values.
            
        Returns:
            dict: Dictionary containing MSE, RMSE, MAE, MAPE, and DirAcc.
        """
        # Ensure numpy arrays for vectorized calculations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 1. TECHNICAL METRICS (The "How Close" questions)
        
        # MSE (Mean Squared Error): Penalizes large errors heavily.
        mse = np.mean((y_true - y_pred) ** 2)
        
        # RMSE (Root MSE): In the same units as the data ($). Easier to interpret.
        rmse = np.sqrt(mse)
        
        # MAE (Mean Absolute Error): The average dollar error. Robust to outliers.
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE (Mean Absolute Percentage Error): Relative error (%).
        # We add a small epsilon to avoid division by zero.
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # 2. BUSINESS METRICS (The "Did we make money?" questions)
        
        # Directional Accuracy (DA):
        # Did the model correctly predict if the price would go UP or DOWN?
        # This is often MORE important for trading strategies than the exact price.
        
        # Calculate changes (diff) from the previous time step
        # Note: We assume y_true and y_pred are aligned sequences. 
        # For a true directional check, we need the *previous* actual value.
        # Since we are predicting 1-step ahead in the rolling window, 
        # we strictly compare the sign of the prediction error vs trend? 
        # No, simpler: sign(y_pred_t - y_true_{t-1}) == sign(y_true_t - y_true_{t-1})
        # However, inside this method we often only receive the vectors. 
        # Simplified proxy for Directional Accuracy: 
        # If we predicted movement X, did it move X?
        
        # For robustness in this static method, we calculate DA only if len > 1
        # otherwise it's impossible to know the trend.
        # Caso 1: Tenemos vector de predicción largo (Horizon > 1)
        if len(y_true) > 1:
            diff_true = np.diff(y_true)
            diff_pred = np.diff(y_pred)
            correct_directions = np.sign(diff_true) == np.sign(diff_pred)
            dir_acc = np.mean(correct_directions) * 100
            
        # Caso 2: Predicción de 1 solo paso (necesitamos el histórico anterior)
        elif len(y_true) == 1 and previous_y is not None:
            true_move = y_true[0] - previous_y
            pred_move = y_pred[0] - previous_y
            
            # Si el signo del movimiento es igual (ambos suben o ambos bajan)
            if np.sign(true_move) == np.sign(pred_move):
                dir_acc = 100.0
            else:
                dir_acc = 0.0
        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "Directional_Accuracy": float(dir_acc)
        }