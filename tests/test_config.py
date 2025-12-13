import pytest
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'experiments.yaml')

def test_yaml_structure():
    """Verifica que el archivo de configuración existe y es legible"""
    assert os.path.exists(CONFIG_PATH), "No se encuentra config/experiments.yaml"
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f: # encoding para evitar problemas con caracteres especiales
        config = yaml.safe_load(f)
        
    assert 'global' in config, "Falta la sección 'global'"
    assert 'models' in config, "Falta la sección 'models'"
    
    # Verificar tipos de datos clave
    assert isinstance(config['global']['input_window_size'], int)
    assert isinstance(config['global']['n_windows'], int)