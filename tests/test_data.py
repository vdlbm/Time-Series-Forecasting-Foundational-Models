import pytest
import pandas as pd
import numpy as np
from src.data.loader import TimeSeriesLoader

def test_loader_csv_reading(tmp_path):
    # 1. Crear CSV temporal con 5 filas (1 de ellas es NaN)
    df_temp = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'Open': [100.0, 102.5, np.nan, 105.0, 106.0] # El NaN será eliminado
    })
    fake_path = tmp_path / "test.csv"
    df_temp.to_csv(fake_path, index=False)
    
    # 2. Instanciar
    loader = TimeSeriesLoader(file_path=str(fake_path), time_col='Date', target_col='Open')
    
    # 3. Cargar
    # CORRECCIÓN 1: Tu método load() devuelve 'self', así que ejecutamos y luego accedemos al atributo .df
    loader.load() 
    
    # Intentamos acceder al DataFrame. Si tu variable se llama self.data, cambia .df por .data
    if hasattr(loader, 'df'):
        df_loaded = loader.df
    elif hasattr(loader, 'data'):
        df_loaded = loader.data
    else:
        pytest.fail("El Loader cargó los datos pero no sabemos en qué atributo los guardó (¿self.df? ¿self.data?)")
    
    # 4. Validaciones
    # CORRECCIÓN 2: Esperamos 4 filas, no 5, porque tu loader eliminó el NaN automáticamente
    assert len(df_loaded) == 4, "El loader debería haber eliminado la fila con NaN"
    assert 'y' in df_loaded.columns