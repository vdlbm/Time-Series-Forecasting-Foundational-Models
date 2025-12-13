import pytest
import pandas as pd
from src.models.factory import ModelFactory

# Definimos los escenarios a probar: (NombreModelo, SeasonLength)
@pytest.mark.parametrize("model_type, season", [
    ('AutoARIMA', 7),
    ('Holt', 1),
    ('SimpleExpSmoothing', 1),
    ('HoltWinters', 7)
])
def test_classical_models_execution(sample_data, model_type, season):
    """
    Prueba que TODOS los modelos clásicos se instancian, entrenan y predicen
    sin errores y respetando el formato de salida.
    """
    horizon = 5
    config = {
        'type': 'classical',
        'model_name': model_type,
        'season_length': season,
        'freq': 'D'
    }

    # 1. Instanciación
    model = ModelFactory.get_model(config)
    
    # 2. Entrenamiento
    model.fit(sample_data)
    
    # 3. Predicción
    pred = model.predict(horizon=horizon)
    
    # --- VALIDACIONES DE CALIDAD ---
    
    # A. Estructura
    assert isinstance(pred, pd.DataFrame), "La salida debe ser un DataFrame"
    assert list(pred.columns) == ['ds', 'y_pred'], f"Columnas incorrectas: {pred.columns}"
    
    # B. Dimensiones
    assert len(pred) == horizon, f"Se pidieron {horizon} pasos, se recibieron {len(pred)}"
    
    # C. Tipos de datos
    assert pd.api.types.is_datetime64_any_dtype(pred['ds']), "La columna 'ds' debe ser fecha"
    assert pd.api.types.is_float_dtype(pred['y_pred']), "La columna 'y_pred' debe ser float"
    
    # D. Lógica temporal
    last_train_date = sample_data['ds'].iloc[-1]
    first_pred_date = pred['ds'].iloc[0]
    assert first_pred_date > last_train_date, "La predicción debe ser futura respecto al entrenamiento"