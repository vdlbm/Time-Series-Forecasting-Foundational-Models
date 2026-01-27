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
    print("\n--- INICIANDO EXPERIMENTO TFG (FIXED LOADER VERSION) ---")
    
    # 1. CARGA DE CONFIGURACIÓN
    with open("config/experiments.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    global_cfg = cfg['global']
    models_cfg = cfg['models']
    
    # Listas para almacenar datos (Logs Globales para CSV)
    summary_log: List[Dict] = []
    detailed_predictions_log: List[Dict] = []

    # 2. BUCLE DE DATASETS
    for ds_config in global_cfg['datasets']:
        dataset_name = ds_config['name']
        print(f"\n████████ PROCESANDO DATASET: {dataset_name} ████████")
        
        # Lista temporal para la tabla de consola de ESTE dataset
        dataset_console_table = []
        
        try:
            # A. Carga de Datos
            loader = TimeSeriesLoader(
                file_path=ds_config['path'],
                time_col=global_cfg['time_col'],
                target_col=ds_config['target_col']
            )
            
            # 1. Cargamos el objeto
            loaded_data = loader.load()
            
            # --- CORRECCIÓN DEL ERROR CRÍTICO ---
            # Si el loader devuelve el objeto wrapper (self) en lugar del DataFrame,
            # extraemos el DataFrame del atributo .df (estándar común)
            if isinstance(loaded_data, pd.DataFrame):
                df = loaded_data
            else:
                # Si no es un DataFrame, asumimos que es el loader y tiene los datos en .df
                if hasattr(loaded_data, 'df'):
                    df = loaded_data.df
                else:
                    raise ValueError(f"El loader devolvió un tipo {type(loaded_data)} y no se encuentra el atributo .df")
            
            # 2. Asegurar Índice
            if 'ds' in df.columns:
                df = df.set_index('ds')
            
            # 3. Resampleo y Lógica TimeGPT (Sanitización)
            # Usamos 'last' para precios de cierre
            df = df.resample(ds_config['frequency']).last()
            
            # Forward Fill: Vital para no tener huecos en TimeGPT
            df = df.asfreq(ds_config['frequency'])
            df = df.ffill()
            
            # 4. Filtros de Fecha (TFG Optimización)
            # Corte estricto final
            df = df[df.index < '2026-01-01']
            
            # Corte inicial según frecuencia
            if ds_config['frequency'] == 'D':
                df = df[df.index >= '2020-01-01'] 
            elif ds_config['frequency'] == 'M':
                df = df[df.index >= '2000-01-01']
                
            # Limpieza final de seguridad (por si el principio tiene NaNs)
            df = df.dropna()
            # ----------------------------------------------------
            
            print(f"   -> Datos listos: {len(df)} registros ({ds_config['frequency']}).")
            
            # Validación
            min_required = global_cfg['input_window_size'] + global_cfg['n_windows']
            if len(df) < min_required:
                print(f"   [!] ERROR: Insuficientes datos ({len(df)} < {min_required}). Saltando.")
                continue

            splitter = RollingWindowSplitter(
                n_windows=global_cfg['n_windows'], 
                test_horizon=global_cfg['test_horizon'],
                input_window_size=global_cfg['input_window_size']
            )
            
            # 3. BUCLE DE MODELOS
            for model_key, model_config in models_cfg.items():
                print(f"   > Modelo: {model_key.upper():<15} | Estado: ⏳ Procesando...", end="\r")
                
                # Configuración dinámica
                if model_config['type'] == 'classical':
                    model_config['season_length'] = ds_config['season_length']
                if model_config['type'] == 'llm_local':
                    model_config['dataset_description'] = ds_config.get('description', 'Financial Time Series')

                try:
                    model = ModelFactory.get_model(model_config)
                    model_metrics_accum = []
                    
                    # 4. BUCLE BACKTESTING (Silencioso)
                    for i, (train_df, test_df) in enumerate(splitter.split(df)):
                        
                        # --- FIT & PREDICT ---
                        start_time = time.time()
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
                        
                        # Logs
                        detailed_predictions_log.append({
                            'Dataset': dataset_name,
                            'Frequency': ds_config['frequency'],
                            'Model': model_key,
                            'Type': model_config['type'],
                            'Date': test_df.index[0],
                            'y_true': float(y_true[0]),
                            'y_pred': float(y_pred[0]),
                            **metrics 
                        })
                        
                        metrics['inference_time'] = inference_time
                        model_metrics_accum.append(metrics)

                    # Fin del modelo: Calculamos medias
                    if model_metrics_accum:
                        avg_metrics = pd.DataFrame(model_metrics_accum).mean().to_dict()
                        
                        summary_log.append({
                            'Dataset': dataset_name,
                            'Model': model_key,
                            'Type': model_config['type'],
                            **avg_metrics
                        })
                        
                        dataset_console_table.append({
                            'Model': model_key.upper(),
                            'RMSE': avg_metrics['RMSE'],
                            'Time(s)': avg_metrics['inference_time']
                        })
                        
                        print(f"   > Modelo: {model_key.upper():<15} | Estado: ✅ (RMSE: {avg_metrics['RMSE']:.4f})")
                    else:
                        print(f"   > Modelo: {model_key.upper():<15} | Estado: ❌ (Sin datos)")

                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"   > Modelo: {model_key.upper():<15} | Estado: ⚠️ ERROR ({str(e)})")
                    # print(e) # Descomentar para debug profundo
                    continue
            
            # --- IMPRESIÓN DE TABLA RESUMEN POR DATASET ---
            if dataset_console_table:
                print("\n" + "="*55)
                print(f" RESUMEN DE RENDIMIENTO: {dataset_name}")
                print("="*55)
                df_table = pd.DataFrame(dataset_console_table)
                df_table = df_table[['Model', 'RMSE', 'Time(s)']]
                print(df_table.sort_values('RMSE').to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
                print("="*55 + "\n")
                    
        except Exception as e:
            print(f"\n[!] ERROR CRÍTICO EN DATASET {dataset_name}: {str(e)}")
            continue

    # 5. GUARDADO FINAL
    print("\n--- GUARDANDO RESULTADOS EN DISCO ---")
    if not os.path.exists("results"):
        os.makedirs("results")
        
    if summary_log:
        summary_df = pd.DataFrame(summary_log)
        cols = ['Dataset', 'Model', 'RMSE', 'MAPE', 'Directional_Accuracy', 'inference_time']
        exist_cols = [c for c in cols if c in summary_df.columns] + [c for c in summary_df.columns if c not in cols]
        summary_df[exist_cols].to_csv(f"results/summary_metrics.csv", index=False)
        print("-> 'results/summary_metrics.csv' guardado.")

    if detailed_predictions_log:
        pd.DataFrame(detailed_predictions_log).to_csv(f"results/detailed_predictions.csv", index=False)
        print("-> 'results/detailed_predictions.csv' guardado.")
        
    print("\n--- EJECUCIÓN FINALIZADA ---")

if __name__ == "__main__":
    main()