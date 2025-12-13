import pytest
import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
from src.models.factory import ModelFactory

# Cargar variables de entorno (.env) para TimeGPT
load_dotenv()

# --- DETECCIÓN DE LIBRERÍAS (Para saltar tests si faltan) ---
try:
    import uni2ts
    MOIRAI_AVAILABLE = True
except ImportError:
    MOIRAI_AVAILABLE = False

try:
    # Intentamos importar algo específico de Chronos o transformers
    import transformers
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False


# --- 1. TEST CHRONOS (Amazon) ---
@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="Librería 'transformers' o 'chronos' no instalada.")
def test_chronos_tiny_execution(sample_data):
    """
    Prueba con Chronos-Tiny (versión ligera).
    Verifica que se descarga, tokeniza y predice.
    """
    horizon = 2
    config = {
        'type': 'foundation',
        'model_name': 'Chronos',
        'model_path': 'amazon/chronos-t5-tiny', # 'tiny' es rápido para tests
        'freq': 'D'
    }

    try:
        model = ModelFactory.get_model(config)
    except Exception as e:
        pytest.fail(f"Fallo al cargar Chronos: {e}")

    # Fit (Dummy) & Predict
    model.fit(sample_data)
    pred = model.predict(horizon=horizon)
    
    # Validaciones
    assert len(pred) == horizon, f"Se esperaban {horizon} predicciones"
    assert 'y_pred' in pred.columns
    # Verificar que no devuelva valores nulos
    assert not pred['y_pred'].isnull().any()
    # Verificar tipo de dato (float)
    assert pd.api.types.is_float_dtype(pred['y_pred'])


# --- 2. TEST TIMEGPT (Nixtla) ---
@pytest.mark.skipif(not os.getenv("NIXTLA_API_KEY"), reason="Falta NIXTLA_API_KEY en .env")
def test_timegpt_initialization():
    """
    Verifica conexión con API de Nixtla.
    NO ejecuta predicción para no gastar créditos/dinero.
    """
    config = {'type': 'foundation', 'model_name': 'TimeGPT', 'freq': 'D'}
    
    try:
        model = ModelFactory.get_model(config)
        # Verificamos que el cliente interno existe y tiene la key cargada
        assert model.client is not None
        assert model.client.api_key is not None
    except Exception as e:
        pytest.fail(f"TimeGPT falló al inicializar: {e}")


# --- 3. TEST MOIRAI (Salesforce) ---
@pytest.mark.skipif(not MOIRAI_AVAILABLE, reason="Librería 'uni2ts' (Moirai) no instalada.")
def test_moirai_small_execution(sample_data):
    """
    Prueba End-to-End para Salesforce Moirai (versión Small).
    Verifica descarga de pesos, instanciación y predicción probabilística.
    """
    horizon = 2
    config = {
        'type': 'foundation',
        'model_name': 'Moirai',
        'size': 'small',            # 'small' para descarga rápida
        'test_horizon': horizon,
        'context_length': 20,       # Contexto corto para test
        'patch_size': 'auto',
        'batch_size': 2,            # Bajo para no saturar CPU/GPU en test
        'freq': 'D'
    }

    # 1. Instanciación
    try:
        model = ModelFactory.get_model(config)
    except Exception as e:
        pytest.fail(f"Fallo al instanciar Moirai: {str(e)}")

    assert model is not None

    # 2. Fit & Predict
    model.fit(sample_data)
    
    try:
        pred = model.predict(horizon=horizon)
    except Exception as e:
        pytest.fail(f"Fallo durante Moirai predict(): {str(e)}")

    # 3. Validaciones
    assert len(pred) == horizon
    assert 'y_pred' in pred.columns
    
    # Moirai devuelve la media de la distribución, debe ser un número válido
    assert not pred['y_pred'].isnull().any()
    
    # Sanity Check numérico:
    # La predicción debe ser un número positivo (para precios) y razonable
    val = pred['y_pred'].iloc[0]
    assert isinstance(val, (float, np.floating))
    assert val > 0, "La predicción de Moirai fue negativa o cero"