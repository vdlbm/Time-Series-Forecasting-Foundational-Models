import os
import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient

# 1. Cargar API Key
load_dotenv()
api_key = os.getenv("NIXTLA_API_KEY")
if not api_key:
    print("❌ ERROR: No se encontró NIXTLA_API_KEY en .env")
    exit()

print(f"✅ API Key encontrada: {api_key[:5]}...{api_key[-5:]}")

# 2. Cargar Datos (SP500 Monthly) simulando lo que hace main.py
print("\n--- CARGANDO Y LIMPIANDO DATOS ---")
df = pd.read_csv("data/SP500_Monthly.csv")

# Renombrar a ds/y si es necesario
if 'Date' in df.columns:
    df = df.rename(columns={'Date': 'ds'})
if 'Close' in df.columns:
    df = df.rename(columns={'Close': 'y'})

df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
df = df.set_index('ds').sort_index()

# Simulamos la "Sanitización Paranoica" de main.py
df.index = df.index.map(lambda t: t.replace(day=1)) # Forzar inicio de mes
df = df.resample('MS').last().ffill() # Sin huecos
df = df[df.index < '2026-01-01']
df = df[df.index >= '2000-01-01']
df = df.dropna()

# Preparamos una ventana de entrenamiento (últimos 60 meses)
train_df = df.iloc[-60:].copy().reset_index()
train_df['unique_id'] = 'SP500' # Vital para TimeGPT

print(f"Datos de entrenamiento (Head):\n{train_df.head(3)}")
print(f"Datos de entrenamiento (Tail):\n{train_df.tail(3)}")
print(f"Tamaño: {len(train_df)} filas.")

# 3. LLAMADA DIRECTA A LA API (SIN WRAPPERS)
print("\n--- INVOCANDO A TIMEGPT (NIxtlaClient) ---")
client = NixtlaClient(api_key=api_key)

try:
    # Intento de pronóstico
    fcst_df = client.forecast(
        df=train_df,
        h=1,             # Pronosticar 1 mes
        freq='MS',       # Frecuencia explícita
        time_col='ds',
        target_col='y'
    )
    
    print("\n✅ ÉXITO: La API respondió.")
    print("Columnas devueltas:", fcst_df.columns.tolist())
    print("Contenido:\n", fcst_df)
    
    # Verificación de nombres de columna (El error común)
    if 'TimeGPT' not in fcst_df.columns:
        print("\n⚠️ ALERTA: No veo la columna 'TimeGPT'. Puede que se llame diferente.")
        
except Exception as e:
    print("\n❌ ERROR FATAL DE TIMEGPT:")
    print(e)
    # Si es error de validación, suele traer detalles extra
    if hasattr(e, 'response'):
        print("Respuesta del servidor:", e.response)