import pytest
import numpy as np
from src.evaluation.metrics import PerformanceEvaluator

class TestPerformanceEvaluator:
    
    def test_perfect_prediction(self):
        """Caso: Predicción exacta. El error debe ser 0 y ROI positivo."""
        y_true = np.array([110])
        y_pred = np.array([110])
        previous_y = 100.0  # El precio subió de 100 a 110 (+10%)
        
        metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred, previous_y)
        
        # Métricas Técnicas
        assert metrics['MSE'] == 0.0
        assert metrics['MAE'] == 0.0
        assert metrics['MAPE'] == 0.0
        
        # Métricas de Negocio
        # Dirección: Real (+10), Pred (+10) -> Acierto
        assert metrics['Directional_Accuracy'] == 100.0
        
        # ROI: Compramos (Long) y subió un 10%
        # Asset Return = (110-100)/100 = 0.10
        # Position = 1 (Long)
        # Strategy = 1 * 0.10 * 100 = 10%
        assert metrics['Strategy_Return_Pct'] == 10.0

    def test_wrong_direction_loss(self):
        """Caso: El mercado cae, pero el modelo predijo subida (Pérdida)."""
        y_true = np.array([90])   # Bajó a 90
        y_pred = np.array([105])  # Modelo predijo subida a 105
        previous_y = 100.0
        
        metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred, previous_y)
        
        # Dirección: Real (-10), Pred (+5) -> Fallo
        assert metrics['Directional_Accuracy'] == 0.0
        
        # ROI:
        # Asset Return = (90-100)/100 = -0.10 (-10%)
        # Position = 1 (Long) porque modelo dijo SUBE
        # Strategy = 1 * -0.10 * 100 = -10.0%
        assert metrics['Strategy_Return_Pct'] == -10.0

    def test_short_selling_profit(self):
        """Caso: El mercado cae y el modelo predice caída (Ganancia en Short)."""
        y_true = np.array([90])  # Bajó
        y_pred = np.array([95])  # Modelo predijo bajada
        previous_y = 100.0
        
        metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred, previous_y)
        
        # Dirección: Real (-10), Pred (-5) -> Acierto
        assert metrics['Directional_Accuracy'] == 100.0
        
        # ROI:
        # Asset Return = -0.10
        # Position = -1 (Short)
        # Strategy = (-1) * (-0.10) * 100 = +10.0%
        assert metrics['Strategy_Return_Pct'] == 10.0

    def test_input_shapes(self):
        """Verificar que funciona con listas planas y arrays anidados."""
        y_true = [105]
        y_pred = [[102]] # Lista anidada
        previous_y = 100.0
        
        metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred, previous_y)
        
        assert metrics['MAE'] == 3.0 # |105 - 102|
        assert isinstance(metrics['MSE'], float)

    def test_mape_epsilon(self):
        """Verificar que MAPE no explota si y_true es 0."""
        y_true = np.array([0])
        y_pred = np.array([1])
        previous_y = 0.0
        
        metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred, previous_y)
        
        # El MAPE debería ser grande pero no NaN/Inf gracias al epsilon
        assert metrics['MAPE'] > 0
        assert np.isfinite(metrics['MAPE'])