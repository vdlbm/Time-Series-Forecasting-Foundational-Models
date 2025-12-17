import yaml
import pandas as pd
import time
from typing import List, Dict

# Importamos nuestros módulos (La maquinaria que hemos construido)
from src.data.loader import TimeSeriesLoader
from src.models.factory import ModelFactory
from src.evaluation.splitter import RollingWindowSplitter
from src.evaluation.metrics import PerformanceEvaluator

def main():
    # 1. CARGA DE CONFIGURACIÓN
    print("\n--- INICIANDO EL EXPERIMENTO TFG ---")
    print("Leyendo configuración...")
    
    with open("config/experiments.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    global_cfg = cfg['global']
    models_cfg = cfg['models']

    # 2. INGESTA DE DATOS (ETL)
    print(f"Cargando datos desde {global_cfg['data_path']}...")
    
    # Instanciamos el Loader con la configuración agnóstica
    loader = TimeSeriesLoader(
        file_path=global_cfg['data_path'],
        time_col=global_cfg['time_col'],
        target_col=global_cfg['target_col']
    )
    
    # Cargamos y limpiamos
    # Nota: Si usas Llama-3, asegurate de que los datos tengan la frecuencia correcta
    df = loader.load().resample(
        freq=global_cfg['frequency'],
        agg_method=global_cfg['agg_method']
    )
    
    # 3. CONFIGURACIÓN DEL EVALUADOR (Backtesting Riguroso)
    print(f"Configurando Rolling Window (Ventanas: {global_cfg['n_windows']})...")
    
    splitter = RollingWindowSplitter(
        n_windows=global_cfg['n_windows'], 
        test_horizon=global_cfg['test_horizon'],
        input_window_size=global_cfg['input_window_size']
    )

    # Lista para guardar los resultados de cada iteración (Log)
    results_log: List[Dict] = []

    # 4. BUCLE DE MODELOS (Iteramos sobre el arsenal)
    # models_cfg.items() nos da pares ("arima", {config}), ("llama3", {config})...
    for model_key, model_config in models_cfg.items():
        
        print(f"\n>>> ENTRENANDO MODELO: {model_key.upper()} ({model_config['type']})")
        
        try:
            # A. INSTANCIACIÓN (Factory Pattern)
            # Aquí es donde ocurre la magia: El Factory decide qué clase crear
            model = ModelFactory.get_model(model_config)
            
            # 5. BUCLE DE ROLLING WINDOW (La simulación temporal)
            # Iteramos a través de los cortes temporales generados por el Splitter
            for i, (train_df, test_df) in enumerate(splitter.split(df)):
                
                window_id = i + 1
                #test_date = test_df.index[0] # La fecha que estamos intentando predecir
                test_date = pd.to_datetime(test_df['ds'].iloc[0])
                
                print(f"    [Ventana {window_id}/{global_cfg['n_windows']}] Prediciendo para: {test_date.date()}...", end="\r")
                
                start_time = time.time()
                
                # B. FIT (Actualizar contexto)
                # Arima calcula matrices. Llama-3 actualiza su prompt.
                model.fit(train_df)
                
                # C. PREDICT (Generar futuro)
                forecast_df = model.predict(horizon=len(test_df))
                
                # Medir tiempo de inferencia (dato útil para el TFG)
                inference_time = time.time() - start_time
                
                # D. EVALUATE (Juzgar resultados)
                # Extraemos los arrays numpy para comparar
                y_true = test_df['y'].values
                y_pred = forecast_df['y_pred'].values
                
                # Calculamos métricas (RMSE, ROI, Directional Accuracy)
                metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred)
                
                # E. LOGGING (Guardar evidencia)
                log_entry = {
                    'model': model_key,
                    'type': model_config['type'],
                    'window_id': window_id,
                    'test_date': test_date,
                    'y_true': y_true[0],     # Guardamos el valor real (para gráficas)
                    'y_pred': y_pred[0],     # Guardamos el valor predicho
                    'inference_time': inference_time,
                    **metrics                # Desempaquetamos el diccionario de métricas (MSE, MAE...)
                }
                results_log.append(log_entry)
            
            print(f"\n    Modelo {model_key} completado con éxito.")
                
        except Exception as e:
            # MANEJO DE ERRORES ROBUSTO
            # Si un modelo falla (ej. TimeGPT sin internet), lo registramos y seguimos con el siguiente.
            # No queremos que un error tire 10 horas de experimento.
            print(f"\n  ERROR CRÍTICO en {model_key}: {str(e)}")
            continue

    # 6. GUARDADO DE RESULTADOS FINAL
    print("\n--- EXPERIMENTO FINALIZADO ---")
    print("Guardando resultados...")
    
    if results_log:
        results_df = pd.DataFrame(results_log)
        
        # Guardar CSV crudo (para Excel/Tableau)
        output_filename = f"results_{global_cfg['project_name']}.csv"
        results_df.to_csv(output_filename, index=False)
        
        # Mostrar resumen rápido en consola
        print("\nRESUMEN DE RENDIMIENTO (Promedio):")
        # Agrupamos por modelo y calculamos la media de las métricas
        summary = results_df.groupby('model')[['RMSE', 'MAE', 'Directional_Accuracy', 'inference_time']].mean()
        print(summary)
        
        print(f"\nResultados detallados guardados en: {output_filename}")
    else:
        print("No se generaron resultados. Revisa los errores.")

if __name__ == "__main__":
    main()