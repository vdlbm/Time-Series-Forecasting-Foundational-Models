from .base import BaseForecaster
from .factory import ModelFactory
from .naive import NaiveForecaster

# Heavy wrappers are imported lazily via ModelFactory.get_model() to avoid
# requiring torch, statsforecast, or llama-cpp-python at import time.
# Import them explicitly only when the dependency is available:
#   from src.models.classical import ClassicalWrapper
#   from src.models.llm import LocalLLMWrapper
#   from src.models.foundation import FoundationWrapper

__all__ = [
    'BaseForecaster', 'ModelFactory', 'NaiveForecaster',
]
