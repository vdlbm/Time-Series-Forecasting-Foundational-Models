import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, Any, Optional

# Imports condicionales
try:
    from nixtla import NixtlaClient
except ImportError:
    NixtlaClient = None

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.pandas import PandasDataset
except ImportError:
    MoiraiForecast = None

# Para Chronos
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import BaseForecaster

class FoundationWrapper(BaseForecaster):
    """
    Wrapper unificado para modelos fundacionales con corrección de escala.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'TimeGPT')
        self.freq = config.get('freq', 'D')
        self.last_train_df = None
        
        # Variables para escalado manual (Moirai)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        
        # --- TIMEGPT SETUP ---
        if self.model_name == 'TimeGPT':
            if NixtlaClient is None:
                raise ImportError("Please install 'nixtla'.")
            api_key = os.getenv("NIXTLA_API_KEY")
            self.client = NixtlaClient(api_key=api_key)

        # --- MOIRAI SETUP ---
        elif self.model_name == 'Moirai':
            if MoiraiForecast is None:
                raise ImportError("Please install 'uni2ts' and 'gluonts'.")
            
            size = config.get('size', 'small')
            self.prediction_length = config.get('test_horizon', 1)
            self.context_length = config.get('context_length', 90)
            self.patch_size = config.get('patch_size', 'auto')
            self.batch_size = config.get('batch_size', 32)
            
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
            model_path = config.get('model_path', "amazon/chronos-t5-small")
            print(f"Loading Chronos ({model_path})...")
            try:
                from chronos import ChronosPipeline
                self.pipeline = ChronosPipeline.from_pretrained(
                    model_path,
                    device_map="auto",
                    dtype=torch.bfloat16,
                )
            except ImportError:
                 raise ImportError("Install 'chronos-forecasting' library.")
            print("Chronos Loaded.")
        else:
            raise ValueError(f"Unknown Foundation Model: {self.model_name}")

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Calcula estadísticas para escalado y guarda los datos.
        """
        self.last_train_df = df_train.copy()
        
        # Calculamos estadísticas para estandarización manual (Crítico para Moirai)
        self.scaler_mean = df_train['y'].mean()
        self.scaler_std = df_train['y'].std()
        
        # Evitar división por cero si es una línea plana
        if self.scaler_std == 0:
            self.scaler_std = 1.0

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.last_train_df is None:
            raise ValueError("Fit must be called before predict.")
            
        # --- TIMEGPT ---
        if self.model_name == 'TimeGPT':
            fcst_df = self.client.forecast(
                df=self.last_train_df,
                h=horizon,
                freq=self.freq,
                time_col='ds',
                target_col='y'
            )
            return fcst_df[['ds', 'TimeGPT']].rename(columns={'TimeGPT': 'y_pred'})

        # --- MOIRAI (Con Escalado Manual) ---
        elif self.model_name == 'Moirai':
            # 1. Escalar entrada: (Valor - Media) / Desv
            df_scaled = self.last_train_df.copy()
            df_scaled['y'] = (df_scaled['y'] - self.scaler_mean) / self.scaler_std
            
            # 2. Preparar Dataset GluonTS
            df_moirai = df_scaled.set_index('ds')
            df_moirai = df_moirai.asfreq(self.freq)
            
            # --- FIX: PARCHE PARA GLUONTS/PANDAS ('MS' BUG) ---
            # Pandas moderno lanza error si pasamos 'MS' a to_period().
            # Mapeamos 'MS' (Inicio Mes) a 'M' (Mensual Genérico) solo para GluonTS.
            # Los datos siguen estando en el día 1, pero GluonTS deja de quejarse.
            gluonts_freq = self.freq
            if gluonts_freq == 'MS':
                gluonts_freq = 'M'
            
            ds = PandasDataset(dict(df_moirai), freq=gluonts_freq)
            
            # 3. Predecir
            forecast_it = self.predictor.predict(ds)
            forecast = next(forecast_it)
            
            # 4. Obtener media (en escala normalizada)
            y_pred_scaled = forecast.mean[:horizon]
            
            # 5. Des-escalar salida: (Pred * Desv) + Media
            y_pred_values = (y_pred_scaled * self.scaler_std) + self.scaler_mean
            
            # Fechas (Usamos la freq original para mantener coherencia)
            last_date = self.last_train_df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=self.freq)[1:]
            
            return pd.DataFrame({'ds': future_dates, 'y_pred': y_pred_values})

        # --- CHRONOS (Con Ensemble) ---
        elif self.model_name == 'Chronos':
            # Chronos maneja su propio escalado interno, pero aumentamos las muestras
            context_tensor = torch.tensor(self.last_train_df["y"].values)
            
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=horizon,
                num_samples=20, 
            )
            
            # Media de las 20 muestras
            y_pred_values = forecast[0].mean(dim=0).numpy()
            
            last_date = self.last_train_df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=self.freq)[1:]
            
            return pd.DataFrame({'ds': future_dates, 'y_pred': y_pred_values})