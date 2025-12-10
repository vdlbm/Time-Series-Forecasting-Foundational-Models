from typing import Dict, Any
from .base import BaseForecaster

# Import specific model wrappers
from .classical import ClassicalWrapper
from .llm import LocalLLMWrapper
from .foundation import FoundationWrapper

class ModelFactory:
    """
    Factory Class (Design Pattern).
    Responsible for instantiating the correct model class based on the configuration string.
    This decouples the main execution logic from specific model implementations.
    """
    
    @staticmethod
    def get_model(config: Dict[str, Any]) -> BaseForecaster:
        """
        Creates and returns a model instance based on the 'type' field in config.
        
        Args:
            config (dict): Configuration dictionary containing 'type' 
                           (e.g., 'classical', 'llm_local', 'foundation').
            
        Returns:
            BaseForecaster: An instance of the requested model.
        """
        # Normalize type to lowercase to avoid case-sensitivity errors
        model_type = config.get('type', '').lower()
        
        # 1. Classical Statistical Models (ARIMA, Holt, ETS)
        if model_type == 'classical':
            return ClassicalWrapper(config)
        
        # 2. Local LLM (Llama-3 via HuggingFace)
        elif model_type == 'llm_local':
            return LocalLLMWrapper(config)
        
        # 3. Foundation Models (TimeGPT, Moirai, Chronos)
        elif model_type == 'foundation':
            return FoundationWrapper(config)
        
        # Error Handling for Typographical Errors in YAML
        else:
            raise ValueError(f"Unknown model type: '{model_type}'. Supported: classical, llm_local, foundation.")