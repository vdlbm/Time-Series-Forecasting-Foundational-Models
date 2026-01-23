import numpy as np
from typing import Dict

class PerformanceEvaluator:
    """
    Computes Technical and Business metrics for Time Series Forecasting.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, previous_y: float) -> Dict[str, float]:
        # Convertir a arrays planos para evitar errores de dimensión
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # --- 1. MÉTRICAS TÉCNICAS ---
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # --- 2. MÉTRICAS DE NEGOCIO (ROI / Directional Accuracy) ---
        
        # A. Directional Accuracy (¿Acertamos la dirección?)
        # Comparamos el movimiento real vs el movimiento predicho desde el punto anterior
        if previous_y is not None:
            true_move = y_true[0] - previous_y
            pred_move = y_pred[0] - previous_y
            
            # Si ambos tienen el mismo signo, es un acierto (1), si no (0)
            is_correct_direction = (np.sign(true_move) == np.sign(pred_move))
            dir_acc = 100.0 if is_correct_direction else 0.0
            
            # B. ROI (Retorno de Inversión Simple)
            # Estrategia: Si el modelo dice SUBE -> Compramos (Long). Si dice BAJA -> Vendemos (Short).
            # Retorno del activo = (Precio_Hoy - Precio_Ayer) / Precio_Ayer
            if previous_y == 0:
                asset_return = 0.0 # Evitar división por cero
            else:
                asset_return = (y_true[0] - previous_y) / previous_y
            # Posición: 1 (Long) o -1 (Short)
            position = np.sign(pred_move) if pred_move != 0 else 0
            
            strategy_return = position * asset_return * 100 # En porcentaje
            
        else:
            dir_acc = 0.0
            strategy_return = 0.0

        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "Directional_Accuracy": float(dir_acc),
            "Strategy_Return_Pct": float(strategy_return)
        }