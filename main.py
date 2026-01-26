import yaml
import pandas as pd
import time
import torch
import gc
import os
from typing import List, Dict

# Importaciones del proyecto
from src.data.loader import TimeSeriesLoader
from src.models.factory import ModelFactory
from src.evaluation.splitter import RollingWindowSplitter
from src.evaluation.metrics import PerformanceEvaluator

def main():
    print("\n--- INICIANDO EXPERIMENTO TFG (FINAL OPTIMIZADO) ---")
    
    # 1. CARGA DE CONFIGURACIÓN
    with open("config/experiments.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    global_cfg = cfg['global']
    models_cfg = cfg['models']
    
    # Listas para almacenar datos (Logs)
    summary_log: List[Dict] = []
    detailed_predictions_log: List[Dict] = []

    # 2. BUCLE DE DATASETS
    for ds_config in global_cfg['datasets']:
        dataset_name = ds_config['name']
        print(f"\n████████ PROCESANDO DATASET: {dataset_name} ████████")
        
        try:
            # A. Carga de Datos
            loader = TimeSeriesLoader(
                file_path=ds_config['path'],
                time_col=global_cfg['time_col'],
                target_col=ds_config['target_col']
            )
            
            # Carga base y resampleo
            df = loader.load().resample(ds_config['frequency']).dropna().set_index('ds')
            
            # --- FILTRADO INTELIGENTE POR FRECUENCIA ---
            if ds_config['frequency'] == 'D':
                # Diario: Solo historia reciente (Post-COVID + un poco antes)
                # ~1000 - 1500 datos
                df = df[df.index >= '2020-01-01'] 
                print(f"   -> [OPTIMIZACIÓN] Diario: Filtrado desde 2020 (foco corto plazo).")
                
            elif ds_config['frequency'] == 'M':
                # Mensual: Historia media (últimos ~25-30 años)
                # Suficiente para ciclos macro, pero evitamos datos del siglo pasado (1920...)
                # ~300 - 400 datos
                df = df[df.index >= '2000-01-01']
                print(f"   -> [OPTIMIZACIÓN] Mensual: Filtrado desde 2000 (foco ciclo moderno).")
            
            print(f"   -> Registros finales para entrenar: {len(df)}. Frecuencia: {ds_config['frequency']}")
            
            # Validación de seguridad
            min_required = global_cfg['input_window_size'] + global_cfg['n_windows']
            if len(df) < min_required:
                print(f"   [!] ERROR BLOQUEANTE: Tienes {len(df)} datos, pero necesitas {min_required}. Saltando dataset...")
                continue

            # B. Configuración Splitter
            splitter = RollingWindowSplitter(
                n_windows=global_cfg['n_windows'], 
                test_horizon=global_cfg['test_horizon'],
                input_window_size=global_cfg['input_window_size']
            )
            
            # 3. BUCLE DE MODELOS
            for model_key, model_config in models_cfg.items():
                print(f"\n   >>> Modelo: {model_key.upper()} ({model_config['type']})")
                
                # Inyección dinámica de estacionalidad
                if model_config['type'] == 'classical':
                    model_config['season_length'] = ds_config['season_length']
                
                # Inyección de descripción semántica para LLMs
                if model_config['type'] == 'llm_local':
                    model_config['dataset_description'] = ds_config.get('description', 'Financial Time Series')

                try:
                    # Instanciación
                    model = ModelFactory.get_model(model_config)
                    
                    # 4. BUCLE DE ROLLING WINDOW (Backtesting)
                    model_metrics_accum = []
                    
                    for i, (train_df, test_df) in enumerate(splitter.split(df)):
                        window_id = i + 1
                        test_date = test_df.index[0]
                        
                        print(f"       [Ventana {window_id}] Prediciendo {test_date.date()}...", end="\r")
                        
                        # --- FIT & PREDICT ---
                        start_time = time.time()
                        
                        # Reset index para que modelos vean 'ds' como columna
                        model.fit(train_df.reset_index())
                        
                        forecast_df = model.predict(horizon=len(test_df))
                        inference_time = time.time() - start_time
                        
                        # --- EVALUACIÓN ---
                        y_true = test_df['y'].values
                        y_pred = forecast_df['y_pred'].values
                        last_known_y = train_df['y'].iloc[-1]
                        
                        metrics = PerformanceEvaluator.calculate_metrics(
                            y_true=y_true, 
                            y_pred=y_pred, 
                            previous_y=last_known_y
                        )
                        
                        # --- LOGGING DETALLADO ---
                        detailed_predictions_log.append({
                            'Dataset': dataset_name,
                            'Frequency': ds_config['frequency'],
                            'Model': model_key,
                            'Type': model_config['type'],
                            'Date': test_date,
                            'y_true': float(y_true[0]),
                            'y_pred': float(y_pred[0]),
                            'Previous_y': float(last_known_y),
                            'Error': float(y_true[0] - y_pred[0]),
                            **metrics 
                        })
                        
                        metrics['inference_time'] = inference_time
                        model_metrics_accum.append(metrics)

                    # Fin del Backtesting para este modelo
                    if model_metrics_accum:
                        avg_metrics = pd.DataFrame(model_metrics_accum).mean().to_dict()
                        
                        summary_log.append({
                            'Dataset': dataset_name,
                            'Model': model_key,
                            'Type': model_config['type'],
                            **avg_metrics
                        })
                        
                        print(f"\n       -> Completado. RMSE Medio: {avg_metrics['RMSE']:.4f}")
                    else:
                        print("\n       -> Sin resultados.")

                    # Limpieza de Memoria
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n       [!] ERROR en {model_key}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"\n[!] ERROR CRÍTICO EN DATASET {dataset_name}: {str(e)}")
            continue

    # 5. GUARDADO DE RESULTADOS FINAL
    print("\n\n--- GUARDANDO RESULTADOS ---")
    
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # A. Tabla Resumen 
    if summary_log:
        summary_df = pd.DataFrame(summary_log)
        desired_cols = ['Dataset', 'Model', 'RMSE', 'MAPE', 'Directional_Accuracy', 'Strategy_Return_Pct', 'inference_time']
        final_cols = [c for c in desired_cols if c in summary_df.columns]
        other_cols = [c for c in summary_df.columns if c not in final_cols]
        final_cols.extend(other_cols)
        
        summary_df[final_cols].to_csv(f"results/summary_metrics.csv", index=False)
        print("-> 'results/summary_metrics.csv' guardado.")

    # B. Predicciones Detalladas
    if detailed_predictions_log:
        detailed_df = pd.DataFrame(detailed_predictions_log)
        detailed_df.to_csv(f"results/detailed_predictions.csv", index=False)
        print("-> 'results/detailed_predictions.csv' guardado.")
        
    print("\n--- EJECUCIÓN FINALIZADA ---")

if __name__ == "__main__":
    main()