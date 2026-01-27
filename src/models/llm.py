import pandas as pd
from typing import Dict, Any, Optional
from llama_cpp import Llama

# Importamos la clase base
from .base import BaseForecaster
from src.data.adapters import DataAdapter

class LocalLLMWrapper(BaseForecaster):
    """
    Wrapper optimizado para Mac (Apple Silicon) usando GGUF y Llama.cpp.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get('model_path', "models/Meta-Llama-3.1-8B-Instruct-Q8_K_M.gguf")
        
        # Aumentamos contexto por seguridad (aunque usaremos 1024 en config)
        self.context_size = config.get('context_window_size', 2048) 
        self.max_tokens = config.get('max_new_tokens', 10)
        
        self.dataset_description = config.get('dataset_description', 'Financial time series data')
        self.df_history: Optional[pd.DataFrame] = None
        
        print(f"Cargando Llama-3 en Mac desde: {self.model_path}...")
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1, 
                n_ctx=self.context_size,
                verbose=False
            )
            print("✅ Modelo cargado en GPU correctamente.")
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            raise e

    def fit(self, df_train: pd.DataFrame) -> None:
        self.df_history = df_train.copy()
        if 'ds' in self.df_history.columns:
            self.df_history['ds'] = pd.to_datetime(self.df_history['ds'])
            self.df_history.set_index('ds', inplace=True)

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.df_history is None:
            raise ValueError("Model must be fit before predicting.")

        # 1. PREPARAR PROMPT (Data -> Text)
        # Usamos tu DataAdapter ("from X increasing to Y...")
        context_text = DataAdapter.to_llm_prompt(self.df_history, window_size=20) 
        
        # --- CAMBIO CRÍTICO: PROMPT "ANTI-SESGO" ---
        # Le decimos qué es el dato, pero le PROHIBIMOS usar conocimiento externo.
        system_msg = (
            "You are a pure mathematical pattern recognition engine. "
            f"Context: The data represents {self.dataset_description}, BUT you must ignore any real-world knowledge about this asset. "
            "Focus EXCLUSIVELY on the numerical trends provided in the history below. "
            f"Task: Extrapolate the mathematical sequence to predict the NEXT {horizon} value(s). "
            "Output ONLY the numerical value(s), separated by commas."
        )
        # -------------------------------------------

        user_msg = f"History sequence:\n{context_text}\n\nPrediction:"
        
        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # --- LÓGICA DE FRENADO ---
        stop_tokens = ["<|eot_id|>", "\n", "History:", "Note:", "Sequence"]
        if horizon == 1:
            stop_tokens.append(",")

        dynamic_max_tokens = horizon * 12
        
        # 2. INFERENCIA
        try:
            output = self.model(
                prompt,
                max_tokens=dynamic_max_tokens,
                stop=stop_tokens,
                echo=False,
                temperature=0.01 # Determinista al máximo
            )
            
            # 3. PROCESAR RESPUESTA
            generated_text = output['choices'][0]['text'].strip()
            generated_text = generated_text.rstrip(',')
            pred_values = DataAdapter.parse_llm_output(generated_text)
            
        except Exception as e:
            print(f"⚠️ Error en inferencia LLM: {e}")
            pred_values = []

        # Fallback
        if not pred_values:
            if self.df_history is not None and not self.df_history.empty:
                last_val = self.df_history['y'].iloc[-1]
            else:
                last_val = 0.0
            pred_values = [last_val] * horizon
            
        if len(pred_values) < horizon:
            pred_values += [pred_values[-1]] * (horizon - len(pred_values))
            
        pred_values = pred_values[:horizon]
            
        # Formatear salida
        last_date = self.df_history.index[-1]
        freq = pd.infer_freq(self.df_history.index) or 'D'
        future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        return pd.DataFrame({'ds': future_dates, 'y_pred': pred_values})