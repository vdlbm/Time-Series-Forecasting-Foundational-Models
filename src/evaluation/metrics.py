import numpy as np
from typing import Dict, List

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

    # --- FINANCIAL METRICS (computed over aggregated fold returns) ---

    @staticmethod
    def sharpe_ratio(strategy_returns: np.ndarray, periods_per_year: int) -> float:
        """
        Annualized Sharpe Ratio of the long/short strategy.

        Args:
            strategy_returns: Array of per-fold strategy returns (in %).
            periods_per_year: Annualization factor (12 for monthly, 252 for
                              daily 5d/w, 365 for daily 7d/w).

        Returns:
            Annualized Sharpe Ratio. Returns 0.0 if std is zero.
        """
        r = np.array(strategy_returns, dtype=float)
        if len(r) < 2:
            return 0.0
        std = np.std(r, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(r) / std * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(strategy_returns: np.ndarray) -> float:
        """
        Maximum Drawdown of the cumulative equity curve.

        Args:
            strategy_returns: Array of per-fold strategy returns (in %).

        Returns:
            Maximum drawdown as a positive percentage (e.g., 15.3 means -15.3%).
        """
        r = np.array(strategy_returns, dtype=float) / 100.0  # Convert to decimal
        equity = np.cumprod(1.0 + r)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        return float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    @staticmethod
    def calmar_ratio(
        strategy_returns: np.ndarray, periods_per_year: int
    ) -> float:
        """
        Calmar Ratio = Annualized Return / Max Drawdown.

        Args:
            strategy_returns: Array of per-fold strategy returns (in %).
            periods_per_year: Annualization factor.

        Returns:
            Calmar Ratio. Returns 0.0 if max drawdown is zero.
        """
        r = np.array(strategy_returns, dtype=float) / 100.0
        if len(r) < 1:
            return 0.0

        # Annualized return via geometric mean
        total_return = np.prod(1.0 + r)
        n_periods = len(r)
        ann_return = (total_return ** (periods_per_year / n_periods) - 1.0) * 100.0

        mdd = PerformanceEvaluator.max_drawdown(strategy_returns)
        if mdd == 0:
            return 0.0
        return float(ann_return / mdd)
