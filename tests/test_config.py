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
    assert 'datasets' in config, "Falta la sección 'datasets'"
    assert 'models' in config, "Falta la sección 'models'"
    
    # Verificar claves globales
    assert isinstance(config['global']['start_date'], str)
    assert isinstance(config['global']['initial_train_end'], str)
    assert isinstance(config['global']['random_seed'], int)
    
    # Verificar que hay al menos 11 modelos (naive + 4 classical + 4 foundation + 2 LLM)
    assert len(config['models']) >= 11, f"Se esperan >= 11 modelos, hay {len(config['models'])}"
    
    # Verificar que los LLMs tienen disable_thinking configurado
    for key, model_cfg in config['models'].items():
        if model_cfg.get('type') == 'llm_local':
            assert 'disable_thinking' in model_cfg, f"Modelo LLM '{key}' sin disable_thinking"