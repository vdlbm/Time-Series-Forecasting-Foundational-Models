import os
import yfinance as yf
import pandas as pd
import requests
import io

# --- CONFIGURACIÓN ---
DATA_DIR = "data"

# 1. Activos Financieros (Yahoo Finance)
# Formato: ID_Interno: Ticker_Yahoo
YAHOO_ASSETS = {
    "SP500": "^GSPC",      # Mercado Accionario
    "EURUSD": "EURUSD=X",  # Mercado Divisas
    "BTC": "BTC-USD"       # Mercado Cripto
}

# 2. Datos Macro (FRED - Federal Reserve Economic Data)
# Usamos el IPC de EE.UU. (Consumer Price Index for All Urban Consumers)
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"

def setup_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✅ Carpeta '{DATA_DIR}' creada.")

def download_yahoo_assets():
    print("\n--- 1. DESCARGANDO ACTIVOS FINANCIEROS (Yahoo) ---")
    
    for name, ticker in YAHOO_ASSETS.items():
        print(f"Procesando: {name} ({ticker})...")
        try:
            # A. Descarga
            df = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
            
            if df.empty:
                print(f"  ❌ Error: No hay datos para {name}")
                continue
            
            # Limpieza: Solo nos interesa el Cierre
            # A veces yfinance devuelve MultiIndex, lo aplanamos si es necesario
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            else:
                df = df[['Close']]
                
            df = df.dropna()
            df.index.name = "Date"
            df.columns = ["Close"] # Estandarizamos nombre de columna

            # B. Guardar Versión DIARIA
            path_daily = os.path.join(DATA_DIR, f"{name}_Daily.csv")
            df.to_csv(path_daily)
            print(f"  -> Diario guardado: {path_daily} ({len(df)} filas)")
            
            # C. Guardar Versión MENSUAL (Resampling)
            # 'M' es Month . Usamos .last() para coger el precio de cierre de mes.
            df_monthly = df.resample('M').last().dropna()
            path_monthly = os.path.join(DATA_DIR, f"{name}_Monthly.csv")
            df_monthly.to_csv(path_monthly)
            print(f"  -> Mensual guardado: {path_monthly} ({len(df_monthly)} filas)")
            
        except Exception as e:
            print(f"  ❌ Excepción en {name}: {e}")

def download_macro_data():
    print("\n--- 2. DESCARGANDO DATOS MACRO (Reserva Federal) ---")
    print("Descargando CPI (Inflación EE.UU.)...")
    
    try:
        # Descarga directa del CSV de la FED
        response = requests.get(FRED_CPI_URL)
        if response.status_code == 200:
            csv_data = io.StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_data, index_col=0, parse_dates=True)
            
            # Renombrar columna extraña 'CPIAUCSL' a 'Close' para que tu Loader funcione igual
            df.columns = ['Close']
            df.index.name = "Date"
            
            # El CPI ya es mensual, así que solo guardamos esa versión
            path_cpi = os.path.join(DATA_DIR, "US_CPI_Monthly.csv")
            df.to_csv(path_cpi)
            print(f"  -> Macro Mensual guardado: {path_cpi} ({len(df)} filas)")
            
            # Nota: No generamos diario porque la inflación diaria no existe oficialmente
        else:
            print(f"  ❌ Error conectando con FRED. Status: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error descargando CPI: {e}")

if __name__ == "__main__":
    setup_dir()
    download_yahoo_assets()
    download_macro_data()
    print("\n✅ ¡PROCESO COMPLETO! Ya puedes configurar tu experiments.yaml")