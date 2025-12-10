import pandas as pd
import torch
import os
from typing import Dict, Any, Optional

# Imports condicionales para evitar errores si no se instalan todas las librerías
try:
    from nixtla import NixtlaClient
except ImportError:
    NixtlaClient = None

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
except ImportError:
    MoiraiForecast = None

# Para Chronos usamos la librería 'transformers' estándar que ya tienes
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import BaseForecaster

class FoundationWrapper(BaseForecaster):
    """
    Wrapper unificado para modelos fundacionales de Series Temporales
    (TimeGPT, Moirai, Chronos).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'TimeGPT')
        self.freq = config.get('freq', 'D')
        self.last_train_df = None
        
        # --- TIMEGPT SETUP ---
        if self.model_name == 'TimeGPT':
            if NixtlaClient is None:
                raise ImportError("Please install 'nixtla' library.")
            
            api_key = os.getenv("NIXTLA_API_KEY")
            if not api_key:
                raise ValueError("NIXTLA_API_KEY not found in environment variables.")
            
            self.client = NixtlaClient(api_key=api_key)
            print("TimeGPT Client Initialized.")

        # --- MOIRAI SETUP ---
        elif self.model_name == 'Moirai':
            if MoiraiForecast is None:
                raise ImportError("Please install 'uni2ts' and 'gluonts'.")
            
            size = config.get('size', 'small') # small, base, large
            self.prediction_length = config.get('test_horizon', 1)
            self.context_length = config.get('context_length', 90)
            self.patch_size = config.get('patch_size', 'auto')
            self.batch_size = config.get('batch_size', 32)
            
            # Cargar modelo Moirai desde HuggingFace
            print(f"Loading Moirai ({size})...")
            self.module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{size}")
            self.predictor = MoiraiForecast(
                module=self.module,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).create_predictor(batch_size=self.batch_size)
            print("Moirai Loaded.")

        # --- CHRONOS SETUP ---
        elif self.model_name == 'Chronos':
            # Amazon Chronos usa la arquitectura T5 (Text-to-Text) pero con tokens numéricos
            model_path = config.get('model_path', "amazon/chronos-t5-small")
            print(f"Loading Chronos ({model_path})...")
            
            # Carga usando la pipeline de Chronos (requiere instalar la lib de amazon-chronos o usar transformers puro)
            # Aquí usamos el enfoque puro de Transformers para máxima compatibilidad
            # Nota: Chronos requiere una lógica específica de tokenización, para simplificar 
            # asumiremos que el usuario ha instalado 'chronos-forecasting'
            try:
                from chronos import ChronosPipeline
                self.pipeline = ChronosPipeline.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            except ImportError:
                 raise ImportError("Please install 'chronos-forecasting' (pip install git+https://github.com/amazon-science/chronos-forecasting.git)")
            
            print("Chronos Loaded.")

        else:
            raise ValueError(f"Unknown Foundation Model: {self.model_name}")

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Estos modelos son Zero-Shot, no entrenan.
        Solo guardamos los datos para usarlos como contexto en predict().
        """
        self.last_train_df = df_train.copy()

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.last_train_df is None:
            raise ValueError("Fit must be called before predict.")
            
        # --- TIMEGPT INFERENCE ---
        if self.model_name == 'TimeGPT':
            # TimeGPT requiere 'ds', 'y' (ya lo tenemos)
            # Ojo: TimeGPT espera un DataFrame con fecha y valor
            # La librería de Nixtla gestiona la inferencia remota
            fcst_df = self.client.forecast(
                df=self.last_train_df,
                h=horizon,
                freq=self.freq,
                time_col='ds',
                target_col='y'
            )
            # Renombrar columna 'TimeGPT' a 'y_pred'
            return fcst_df[['ds', 'TimeGPT']].rename(columns={'TimeGPT': 'y_pred'})

        # --- MOIRAI INFERENCE ---
        elif self.model_name == 'Moirai':
            # Convertir a GluonTS Dataset
            ds = PandasDataset(dict(self.last_train_df.set_index('ds')), freq=self.freq)
            
            # Generar predicción
            forecast_it = self.predictor.predict(ds)
            forecast = next(forecast_it)
            
            # Extraer la media o mediana de las muestras probabilísticas
            y_pred_values = forecast.mean[:horizon] # O forecast.median
            
            # Construir fechas futuras
            last_date = self.last_train_df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=self.freq)[1:]
            
            return pd.DataFrame({'ds': future_dates, 'y_pred': y_pred_values})

        # --- CHRONOS INFERENCE ---
        elif self.model_name == 'Chronos':
            # Chronos requiere un tensor de torch con los valores históricos
            context_tensor = torch.tensor(self.last_train_df["y"].values)
            
            # Inferencia
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=horizon,
                num_samples=1, # Determinista (o media de muestras)
            )
            
            # forecast es un tensor [num_series, num_samples, horizon]
            # Sacamos el valor medio (o unico si num_samples=1)
            y_pred_values = forecast[0].mean(dim=0).numpy()
            
            # Construir fechas futuras
            last_date = self.last_train_df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=self.freq)[1:]
            
            return pd.DataFrame({'ds': future_dates, 'y_pred': y_pred_values})