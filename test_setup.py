from src.models.factory import ModelFactory
import pandas as pd

print(">>> TEST 2: Modelo Clásico (ARIMA)")
try:
    # Configuración falsa solo para probar
    dummy_config = {
        'type': 'classical',
        'model_name': 'AutoARIMA',
        'season_length': 7,
        'freq': 'D'
    }
    
    model = ModelFactory.get_model(dummy_config)
    print(f"✅ Modelo creado: {model}")
    
    # Prueba de entrenamiento falsa
    print("   Intentando fit simulado...")
    df_fake = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'y': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    model.fit(df_fake)
    pred = model.predict(horizon=1)
    print(f"✅ Predicción de prueba: {pred['y_pred'].values[0]}")
    
except Exception as e:
    print(f"❌ ERROR en ARIMA: {e}")