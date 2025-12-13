import pytest
import numpy as np
from src.evaluation.metrics import PerformanceEvaluator

# NOTA: No hace falta sys.path.append gracias a conftest.py

def test_basic_metrics_calculation():
    """
    Prueba de cálculo matemático puro con listas simples.
    """
    # Escenario:
    # Real: 100 -> 200
    # Pred: 110 -> 190 (Error de 10 en ambos casos)
    y_true = [100, 200]
    y_pred = [110, 190]
    
    metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred)
    
    # Validaciones Técnicas
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert metrics["MAPE"] == pytest.approx(7.5, 0.001)

def test_directional_accuracy_logic():
    """
    Verifica la lógica de negocio (subidas y bajadas).
    """
    # CASO: Tendencia Correcta (Ambos suben)
    # Real: 10 -> 20
    # Pred: 10 -> 15
    metrics = PerformanceEvaluator.calculate_metrics([10, 20], [10, 15])
    assert metrics["Directional_Accuracy"] == 100.0

    # CASO: Tendencia Incorrecta (Real sube, Pred baja)
    metrics_fail = PerformanceEvaluator.calculate_metrics([10, 20], [10, 5])
    assert metrics_fail["Directional_Accuracy"] == 0.0

def test_integration_with_pandas_fixture(sample_data):
    """
    AQUÍ REUSAMOS LA FIXTURE DE CONFTEST.
    Probamos que el evaluador acepte Series de Pandas (sample_data['y'])
    y las convierta correctamente a numpy internamente.
    """
    # 1. Obtenemos datos reales del fixture
    y_true = sample_data['y'] # Esto es una Pandas Series
    
    # 2. Creamos una predicción falsa: Siempre es el valor real + 5
    y_pred = y_true + 5
    
    # 3. Calculamos métricas
    metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred)
    
    # 4. Validaciones
    # Como sumamos 5 a todo, el MAE debe ser exactamente 5
    assert metrics["MAE"] == 5.0
    # Al ser una suma constante, la tendencia (subidas/bajadas) se mantiene igual
    # excepto quizás en los puntos de inflexión, pero en una serie lineal será alta.
    assert metrics["Directional_Accuracy"] > 90.0

def test_edge_case_single_value():
    """
    Prueba de robustez con un solo dato.
    """
    metrics = PerformanceEvaluator.calculate_metrics([100], [105])
    assert metrics["MAE"] == 5.0
    assert metrics["Directional_Accuracy"] == 0.0 # No hay tendencia con 1 punto