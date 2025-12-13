import pytest
import pandas as pd
import sys
import os
import torch

# Añadir raíz al path (Para evitar líos de imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def sample_data():
    """
    Fixture que genera un DataFrame de series temporales falso pero realista.
    scope="session" significa que se crea una vez y se reutiliza en todos los tests.
    """
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    # Creamos una serie con tendencia y estacionalidad simple
    values = [100 + i + (10 * (i % 7)) for i in range(50)]
    
    df = pd.DataFrame({'ds': dates, 'y': values})
    return df

@pytest.fixture
def gpu_available():
    """Devuelve True si hay GPU disponible para pruebas pesadas."""
    return torch.cuda.is_available()