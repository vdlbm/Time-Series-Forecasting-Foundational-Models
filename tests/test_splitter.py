import pytest
import pandas as pd
from src.evaluation.splitter import RollingWindowSplitter

# NOTA: Ya no necesitamos sys.path.append aquí

def test_rolling_splitter_mechanics():
    """
    Test matemático estricto. Creamos datos manuales (100 filas) 
    porque sample_data (50 filas) se nos queda corto para esta prueba específica.
    """
    df = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'y': range(100)
    })
    
    splitter = RollingWindowSplitter(n_windows=3, test_horizon=5, input_window_size=50)
    splits = list(splitter.split(df))
    
    assert len(splits) == 3
    assert len(splits[-1][1]) == 5 # Horizonte correcto

def test_no_data_leakage(sample_data):
    """
    AQUÍ SÍ USAMOS LA FIXTURE 'sample_data' del conftest.py.
    Simplemente la pedimos como argumento.
    """
    # Usamos n_windows=1 para simplificar
    splitter = RollingWindowSplitter(n_windows=1, test_horizon=5, input_window_size=20)
    
    for train, test in splitter.split(sample_data):
        max_train_idx = train.index.max()
        min_test_idx = test.index.min()
        
        # Validación
        assert max_train_idx < min_test_idx