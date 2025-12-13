import pytest
import pandas as pd
from src.data.adapters import DataAdapter

def test_dataframe_to_prompt_string():
    """Prueba que convierte una serie temporal en texto legible para Llama"""
    df = pd.DataFrame({'y': [10.5, 20.1, 30.0]}, 
                      index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    
    prompt_text = DataAdapter.to_llm_prompt(df, window_size=3)
    
    assert isinstance(prompt_text, str)
    assert "10.5" in prompt_text
    assert "20.1" in prompt_text
    # Verificamos que no meta índices raros, solo los valores
    assert "y" not in prompt_text 

@pytest.mark.parametrize("llm_output, expected_values", [
    ("The next values are 100.5, 101.2 and 102.0", [100.5, 101.2, 102.0]), # Texto con ruido
    ("100, 200, 300", [100.0, 200.0, 300.0]),                               # Limpio comas
    ("100 200 300", [100.0, 200.0, 300.0]),                                 # Limpio espacios
    ("I think it will go down to 50.5.", [50.5]),                           # Frase compleja
    ("No numbers here", [])                                                 # Caso borde vacio
])
def test_parsing_llm_output(llm_output, expected_values):
    """
    Prueba que el parser es capaz de extraer números de lo que "escupa" el LLM.
    """
    parsed = DataAdapter.parse_llm_output(llm_output)
    assert parsed == expected_values, f"Fallo al parsear: '{llm_output}'"