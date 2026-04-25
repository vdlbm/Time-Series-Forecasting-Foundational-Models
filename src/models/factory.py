from typing import Dict, Any
from .base import BaseForecaster


class ModelFactory:
    """
    Factory Class (Design Pattern).
    Responsible for instantiating the correct model class based on the configuration string.
    This decouples the main execution logic from specific model implementations.
    
    Imports are deferred to each branch so that heavy dependencies (torch,
    statsforecast, llama-cpp-python) are only loaded when actually needed.
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
        
        # 0. Naive Baseline (Random Walk)
        if model_type == 'naive':
            from .naive import NaiveForecaster
            return NaiveForecaster(config)
        
        # 1. Classical Statistical Models (ARIMA, Holt, ETS)
        elif model_type == 'classical':
            from .classical import ClassicalWrapper
            return ClassicalWrapper(config)
        
        # 2. Local LLM (via HuggingFace)
        elif model_type == 'llm_local':
            from .llm import LocalLLMWrapper
            return LocalLLMWrapper(config)
        
        # 3. Foundation Models (TimeGPT, Moirai, Chronos)
        elif model_type == 'foundation':
            from .foundation import FoundationWrapper
            return FoundationWrapper(config)
        
        # Error Handling for Typographical Errors in YAML
        else:
            raise ValueError(f"Unknown model type: '{model_type}'. Supported: naive, classical, llm_local, foundation.")
