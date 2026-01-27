import yaml
import pandas as pd
import time
import torch
import gc
import os
import sys
import logging
import contextlib
from typing import List, Dict
from dotenv import load_dotenv

# Cargar variables de entorno (.env) para TimeGPT
load_dotenv()

# --- SILENCIAR LOGS DE NIXTLA ---
logging.getLogger("nixtla").setLevel(logging.ERROR)

# Importaciones del proyecto
from src.data.loader import TimeSeriesLoader
from src.models.factory import ModelFactory
from src.evaluation.splitter import RollingWindowSplitter
from src.evaluation.metrics import PerformanceEvaluator

# Context manager para silenciar stderr
@contextlib.contextmanager
def suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)

def main():
    print("\n--- INICIANDO EXPERIMENTO TFG (FREQ KEY FIX) ---")
    
    # 1. CARGA DE CONFIGURACIÓN
    with open("config/experiments.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    global_cfg = cfg['global']
    models_cfg = cfg['models']
    
    summary_log: List[Dict] = []
    detailed_predictions_log: List[Dict] = []

    # 2. BUCLE DE DATASETS
    for ds_config in global_cfg['datasets']:
        dataset_name = ds_config['name']
        print(f"\n████████ PROCESANDO DATASET: {dataset_name} ████████")
        
        ds_input_window = ds_config.get('input_window_size', 90)
        ds_test_horizon = ds_config.get('test_horizon', 1)
        ds_n_windows = ds_config.get('n_windows', 12)
        
        print(f"   [Config] Train: {ds_input_window} | Horizon: {ds_test_horizon} | Tests: {ds_n_windows}")
        
        dataset_console_table = []
        
        try:
            # A. Carga de Datos
            loader = TimeSeriesLoader(
                file_path=ds_config['path'],
                time_col=global_cfg['time_col'],
                target_col=ds_config['target_col']
            )
            
            loaded_data = loader.load()
            if isinstance(loaded_data, pd.DataFrame):
                df = loaded_data
            elif hasattr(loaded_data, 'df'):
                df = loaded_data.df
            else:
                raise ValueError("Formato de Loader no reconocido")
            
            if 'ds' in df.columns:
                df = df.set_index('ds')
            
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # --- ESTRATEGIA DUAL ---
            original_target_freq = ds_config['frequency'] 
            calculation_freq = original_target_freq       
            
            if original_target_freq == 'M':
                df.index = df.index.map(lambda t: t.replace(day=1))
                calculation_freq = 'MS'
            
            # Limpieza y Reconstrucción Global
            df = df[~df.index.duplicated(keep='last')]
            df = df.resample(calculation_freq).last()
            
            if len(df) > 1:
                perfect_index = pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq=calculation_freq
                )
                df = df.reindex(perfect_index)
            
            df = df.ffill() 
            
            # Cortes de Fecha
            df = df[df.index < '2026-01-01'] 
            
            if 'D' in calculation_freq:
                df = df[df.index >= '2020-01-01'] 
            elif 'M' in calculation_freq: 
                df = df[df.index >= '2000-01-01']
                
            df = df.dropna()
            df.index.name = 'ds'
            df.index.freq = calculation_freq
            
            print(f"   -> Datos listos: {len(df)} registros. (Calc Freq: {calculation_freq})")
            
            min_required = ds_input_window + (ds_n_windows * ds_test_horizon)
            if len(df) < min_required:
                print(f"   [!] ERROR: Insuficientes datos. Necesarios: {min_required}, Disponibles: {len(df)}")
                continue

            splitter = RollingWindowSplitter(
                n_windows=ds_n_windows, 
                test_horizon=ds_test_horizon,
                input_window_size=ds_input_window
            )
            
            # 3. BUCLE DE MODELOS
            for model_key, model_config in models_cfg.items():
                print(f"   > Modelo: {model_key.upper():<15} | Estado: ⏳ Procesando...", end="\r")
                
                local_model_config = model_config.copy()
                
                # --- CORRECCIÓN CRÍTICA: USAR CLAVE 'freq' ---
                # FoundationWrapper espera 'freq', no 'frequency'
                local_model_config['freq'] = calculation_freq      # <--- ESTA ES LA CLAVE QUE FALTABA
                local_model_config['frequency'] = calculation_freq # Mantenemos esta por si acaso otro modelo la usa
                
                if local_model_config['type'] == 'classical':
                    local_model_config['season_length'] = ds_config['season_length']
                if local_model_config['type'] == 'llm_local':
                    local_model_config['dataset_description'] = ds_config.get('description', 'Financial Time Series')

                try:
                    model = ModelFactory.get_model(local_model_config)
                    model_metrics_accum = []
                    
                    for i, (train_df, test_df) in enumerate(splitter.split(df)):
                        
                        start_time = time.time()
                        
                        # --- SANITIZACIÓN SEGURA ---
                        train_df = train_df.asfreq(calculation_freq)
                        train_df = train_df.ffill().bfill()
                        
                        if train_df.isna().any().any():
                            train_df = train_df.fillna(0)
                        
                        if len(train_df) < (ds_input_window * 0.9): 
                            continue

                        # --- PREPARACIÓN ESPECÍFICA TIMEGPT ---
                        if model_key.lower() == 'timegpt':
                            # Delay para evitar Rate Limit
                            time.sleep(2) 
                            
                            train_df = train_df.reset_index() 
                            train_df['unique_id'] = 'series_1'
                            train_df = train_df[['unique_id', 'ds', 'y']] 
                        else:
                            train_df = train_df.reset_index()

                        # REINTENTOS (RETRY LOGIC)
                        max_retries = 3
                        forecast_df = None
                        last_error = None
                        
                        for attempt in range(max_retries):
                            try:
                                model.fit(train_df) 
                                forecast_df = model.predict(horizon=len(test_df))
                                break 
                            except Exception as e:
                                last_error = e
                                if "429" in str(e):
                                    time.sleep(5 * (attempt + 1))
                                else:
                                    time.sleep(1)
                        
                        if forecast_df is None:
                            print(f"\n   [!] Error fatal en ventana {i} ({model_key})")
                            print(f"       Error: {str(last_error)}")
                            continue
                            
                        inference_time = time.time() - start_time
                        
                        y_true = test_df['y'].values
                        y_pred = forecast_df['y_pred'].values
                        last_known_y = train_df['y'].iloc[-1]
                        
                        metrics = PerformanceEvaluator.calculate_metrics(
                            y_true=y_true, 
                            y_pred=y_pred, 
                            previous_y=last_known_y
                        )
                        
                        # REVERT LOGIC
                        report_date = test_df.index[0]
                        if original_target_freq == 'M':
                            report_date = report_date + pd.offsets.MonthEnd(0)
                        
                        detailed_predictions_log.append({
                            'Dataset': dataset_name,
                            'Frequency': original_target_freq, 
                            'Model': model_key,
                            'Type': local_model_config['type'],
                            'Date': report_date,               
                            'y_true': float(y_true[0]),
                            'y_pred': float(y_pred[0]),
                            **metrics 
                        })
                        
                        metrics['inference_time'] = inference_time
                        model_metrics_accum.append(metrics)

                    if model_metrics_accum:
                        avg_metrics = pd.DataFrame(model_metrics_accum).mean().to_dict()
                        summary_log.append({
                            'Dataset': dataset_name,
                            'Model': model_key,
                            'Type': local_model_config['type'],
                            **avg_metrics
                        })
                        dataset_console_table.append({
                            'Model': model_key.upper(),
                            'RMSE': avg_metrics['RMSE'],
                            'Time(s)': avg_metrics['inference_time']
                        })
                        print(f"   > Modelo: {model_key.upper():<15} | Estado: ✅ (RMSE: {avg_metrics['RMSE']:.4f})")
                    else:
                        print(f"   > Modelo: {model_key.upper():<15} | Estado: ⚠️ (Falló en todas las ventanas)")

                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"   > Modelo: {model_key.upper():<15} | Estado: ❌ ERROR GENERAL ({str(e)[:100]}...)")
                    continue
            
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