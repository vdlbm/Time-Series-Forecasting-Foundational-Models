import torch
import pandas as pd
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base import BaseForecaster
from src.data.adapters import DataAdapter

class LocalLLMWrapper(BaseForecaster):
    """
    Wrapper for running Large Language Models (LLMs) locally.
    Implements 4-bit quantization to allow execution on consumer hardware.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_id = config.get('model_id', "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.context_size = config.get('context_window_size', 90)
        self.max_tokens = config.get('max_new_tokens', 20)
        self.df_history: Optional[pd.DataFrame] = None
        self.quantization_4bit = config.get('quantization_4bit', True)

        # Determine device type: GPU (CUDA) or CPU
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   >>> Iniciando LLM en: {self.device_type}")
        
        # 1. QUANTIZATION CONFIGURATION (Hardware Optimization)
        # We use NF4 (Normal Float 4-bit) which offers the best accuracy/compression ratio.
        bnb_config = None
        if self.quantization_4bit and self.device_type == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        print(f"Loading LLM: {self.model_id}...")
        
        # 2. TOKENIZER LOADING
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        # Llama 3 requires explicit padding token definition for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # 3. MODEL LOADING
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config if config.get('quantization_4bit', True) else None,
            device_map="auto",
            low_cpu_mem_usage=False,
            trust_remote_code=True
        )

        # Ignores the training mode, we only do inference. 2 consequences:
        # Stops dropout layers and neurons from being active
        # Stops batchnorm layers from updating running stats
        #self.model.eval()

        print("Model loaded successfully on GPU/CPU.")

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Updates the context window.
        Unlike ARIMA, we don't 'train' weights here (too slow). 
        We simply store the recent history to build the prompt later.
        """
        self.df_history = df_train.copy()

        # Ensure 'ds' is datetime and set as index
        if 'ds' in self.df_history.columns:
            # Aseguramos que sea tipo fecha
            self.df_history['ds'] = pd.to_datetime(self.df_history['ds'])
            # La movemos al índice
            self.df_history.set_index('ds', inplace=True)

    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generates the forecast using Prompt Engineering + Inference.
        """
        if self.df_history is None:
            raise ValueError("Model must be fit before predicting.")

        # 1. PREPARE PROMPT (Data -> Text)
        # Convert numerical history to narrative text
        context_text = DataAdapter.to_llm_prompt(self.df_history, window_size=self.context_size)
        prompt = self._build_llama3_prompt(context_text, horizon)
        
        # 2. TOKENIZATION
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 3. INFERENCE (Generation)
        # torch.no_grad() is critical to save memory (we don't need gradients for inference)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,    # CRITICAL: Temperature = 0 (Deterministic output)
                # temperature=0.1,  # Only use if do_sample=True
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # 4. DECODING
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the NEWLY generated text (remove the prompt)
        # Since we skipped special tokens, exact string matching is safer
        # But a heuristic approach is to find the last known number or split by header
        # For simplicity and robustness with Llama-3's specific structure:
        raw_prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        generated_part = full_response.replace(raw_prompt_text, "").strip()
        
        # 5. PARSING (Text -> Data)
        pred_values = DataAdapter.parse_llm_output(generated_part)
        
        # 6. FALLBACK MECHANISM (Robustness)
        # If LLM fails to output a number (e.g., outputs "I cannot predict markets"),
        # we fallback to the last known value to prevent pipeline crash.
        if not pred_values:
            print(f"LLM Parsing Warning: Output was '{generated_part}'. Using fallback.")
            last_val = self.df_history['y'].iloc[-1]
            pred_values = [last_val] * horizon
            
        # Ensure we return exactly 'horizon' steps
        if len(pred_values) < horizon:
            pred_values += [pred_values[-1]] * (horizon - len(pred_values))
            
        # 7. FORMAT OUTPUT
        last_date = self.df_history.index[-1]
        freq = pd.infer_freq(self.df_history.index) or 'D'
        future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        return pd.DataFrame({'ds': future_dates, 'y_pred': pred_values[:horizon]})

    def _build_llama3_prompt(self, context_str: str, steps: int) -> str:
        """
        Constructs a prompt strictly adhering to Llama-3 Instruct format.
        Tags: <|begin_of_text|>, <|start_header_id|>, etc.
        """
        system_msg = (
            "You are an expert financial forecaster. "
            "You analyze a sequence of price movements described in text. "
            f"Your task is to predict strictly the NEXT {steps} price(s) following the pattern. "
            "Output ONLY the numerical value(s). Do not provide explanations or disclaimers."
        )
        
        user_msg = f"Here is the price history:\n{context_str}\n\nPredict the next value:"
        
        # Llama-3 Official Prompt Template
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt