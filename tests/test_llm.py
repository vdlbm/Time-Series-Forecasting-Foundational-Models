import pytest
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from src.models.llm import LocalLLMWrapper

# Configuración básica para los tests
CONFIG = {
    "model_path": "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "context_window_size": 100,
    "max_new_tokens": 5
}

class TestLocalLLM:

    @patch('src.models.llm.Llama')
    def test_llm_initialization_mock(self, mock_llama_class):
        """
        Prueba Unitaria: Verifica que el wrapper inicializa la clase Llama correctamente
        sin cargar el modelo pesado en memoria.
        """
        # Configuramos el Mock
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance
        
        # Instanciamos nuestro wrapper
        model = LocalLLMWrapper(CONFIG)
        
        # Verificamos que llamó a Llama con n_gpu_layers=-1 (Metal/GPU activado)
        mock_llama_class.assert_called_once()
        _, kwargs = mock_llama_class.call_args
        assert kwargs['n_gpu_layers'] == -1
        assert kwargs['model_path'] == CONFIG['model_path']

    @patch('src.models.llm.Llama')
    def test_predict_mock_output(self, mock_llama_class):
        """
        Prueba Unitaria: Verifica que el flujo de 'predict' procesa la respuesta.
        """
        # 1. Mockear la respuesta del LLM
        mock_instance = MagicMock()
        # Simulamos que Llama devuelve un diccionario estilo OpenAI
        mock_instance.return_value = {
            'choices': [{'text': ' 105.5, 106.2 '}]
        }
        mock_llama_class.return_value = mock_instance
        
        # 2. Preparar datos dummy
        df_train = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=20),
            'y': range(20)
        }).set_index('ds')
        
        # 3. Ejecutar predicción
        model = LocalLLMWrapper(CONFIG)
        model.fit(df_train)
        prediction = model.predict(horizon=2)
        
        # 4. Validaciones
        assert isinstance(prediction, pd.DataFrame)
        assert len(prediction) == 2
        assert 'y_pred' in prediction.columns
        # Verificar que parseó los números del texto mockeado
        assert prediction['y_pred'].iloc[0] == 105.5

    @pytest.mark.skipif(not os.path.exists(CONFIG['model_path']), reason="Modelo GGUF no encontrado en local")
    def test_real_model_loading(self):
        """
        Prueba de Integración: Intenta cargar el modelo REAL si existe el archivo.
        Verifica que tu Mac es capaz de levantar el GGUF.
        """
        print("\n⚠️ Ejecutando test de carga REAL (esto puede tardar unos segundos)...")
        
        try:
            model = LocalLLMWrapper(CONFIG)
            assert model.model is not None
            print("✅ Modelo cargado exitosamente en memoria.")
        except Exception as e:
            pytest.fail(f"Falló la carga del modelo real: {e}")

    def test_predict_without_fit_raises_error(self):
        """Verifica que lanza error si intentamos predecir sin entrenar (fit)."""
        # Usamos patch para evitar cargar el modelo real aquí también
        with patch('src.models.llm.Llama'):
            model = LocalLLMWrapper(CONFIG)
            with pytest.raises(ValueError, match="Model must be fit"):
                model.predict(horizon=1)