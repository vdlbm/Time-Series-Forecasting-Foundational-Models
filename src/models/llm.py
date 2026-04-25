import re
import pandas as pd
from typing import Dict, Any, Optional
from llama_cpp import Llama

# Importamos la clase base
from .base import BaseForecaster
from src.data.adapters import DataAdapter

class LocalLLMWrapper(BaseForecaster):
    """
    Model-agnostic wrapper for local LLM inference via llama.cpp (GGUF format).

    Uses create_chat_completion() which auto-detects the chat template from
    GGUF metadata, making it compatible with Gemma, Phi-4, Llama, Qwen, etc.

    Handles Qwen3's "thinking mode" by appending /no_think to the user message
    and stripping any residual <think>...</think> content from the output.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get('model_path', "models/model.gguf")
        
        self.context_size = config.get('context_window_size', 4096) 
        self.max_tokens = config.get('max_new_tokens', 20)
        self.llm_window_size = config.get('llm_window_size', 40)
        self.disable_thinking = config.get('disable_thinking', False)
        
        self.dataset_description = config.get('dataset_description', 'Financial time series data')
        self.df_history: Optional[pd.DataFrame] = None
        
        print(f"Loading LLM from: {self.model_path}...")
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1, 
                n_ctx=self.context_size,
                verbose=False
            )
            print(f"✅ {self.name} loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading {self.name}: {e}")
            raise e

    def fit(self, df_train: pd.DataFrame) -> None:
        self.df_history = df_train.copy()
        if 'ds' in self.df_history.columns:
            self.df_history['ds'] = pd.to_datetime(self.df_history['ds'])
            self.df_history.set_index('ds', inplace=True)

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.df_history is None:
            raise ValueError("Model must be fit before predicting.")

        # 1. PREPARE PROMPT (Data -> Text)
        context_text = DataAdapter.to_llm_prompt(
            self.df_history, window_size=self.llm_window_size
        )
        
        # Anti-bias system message: force pure pattern recognition
        system_msg = (
            "You are a pure mathematical pattern recognition engine. "
            f"Context: The data represents {self.dataset_description}, "
            "BUT you must ignore any real-world knowledge about this asset. "
            "Focus EXCLUSIVELY on the numerical trends provided in the history below. "
            f"Task: Extrapolate the mathematical sequence to predict the NEXT {horizon} value(s). "
            "Output ONLY the numerical value(s), separated by commas."
        )

        user_msg = f"History sequence:\n{context_text}\n\nPrediction:"

        # Qwen3 thinking mode: append /no_think to force direct output
        if self.disable_thinking:
            user_msg += " /no_think"

        # Model-agnostic chat completion (template auto-detected from GGUF)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Stop tokens (universal + model-specific safety nets)
        stop_tokens = ["History:", "Note:", "Sequence:", "Based on"]
        if self.disable_thinking:
            stop_tokens.append("<think>")  # Safety net: halt if thinking starts
        if horizon == 1:
            stop_tokens.append(",")

        dynamic_max_tokens = max(horizon * 12, self.max_tokens)
        
        # 2. INFERENCE
        try:
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=dynamic_max_tokens,
                stop=stop_tokens,
                temperature=0.01,  # Near-deterministic
            )
            
            # 3. PARSE RESPONSE
            generated_text = output['choices'][0]['message']['content'].strip()

            # Strip any residual <think>...</think> content (Qwen3 safety)
            generated_text = re.sub(
                r'<think>.*?</think>', '', generated_text, flags=re.DOTALL
            ).strip()

            generated_text = generated_text.rstrip(',')
            pred_values = DataAdapter.parse_llm_output(generated_text)
            
        except Exception as e:
            print(f"⚠️ LLM inference error ({self.name}): {e}")
            pred_values = []

        # Fallback: repeat last known value
        if not pred_values:
            if self.df_history is not None and not self.df_history.empty:
                last_val = self.df_history['y'].iloc[-1]
            else:
                last_val = 0.0
            pred_values = [last_val] * horizon
            
        if len(pred_values) < horizon:
            pred_values += [pred_values[-1]] * (horizon - len(pred_values))
            
        pred_values = pred_values[:horizon]
            
        # Format output
        last_date = self.df_history.index[-1]
        freq = pd.infer_freq(self.df_history.index) or 'D'
        future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        return pd.DataFrame({'ds': future_dates, 'y_pred': pred_values})
