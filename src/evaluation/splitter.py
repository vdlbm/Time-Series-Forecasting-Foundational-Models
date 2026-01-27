import pandas as pd
from typing import Generator, Tuple

class RollingWindowSplitter:
    """
    Generates time-ordered train/test splits for Backtesting.
    
    UPDATED: Implements a 'Sliding Window' strategy.
    - Train set: Fixed size (input_window_size) moving forward.
    - Test set: Fixed size (test_horizon) moving forward.
    """
    
    def __init__(self, n_windows: int, test_horizon: int, input_window_size: int):
        """
        Args:
            n_windows (int): Number of folds to test.
            test_horizon (int): Number of steps to predict (usually 1).
            input_window_size (int): Exact size of the training window.
        """
        self.n_windows = n_windows
        self.test_horizon = test_horizon
        self.input_window_size = input_window_size

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        total_rows = len(df)
        
        # Validar tamaño necesario: Ventana Entreno + (Ventanas Test * Horizonte)
        required_samples = self.input_window_size + (self.n_windows * self.test_horizon)
        
        if total_rows < required_samples:
            raise ValueError(
                f"Dataset too small. Needed {required_samples} rows, "
                f"but got {total_rows}. Reduce n_windows or input_window_size."
            )

        # Calculamos dónde empieza el PRIMER test
        start_test_idx = total_rows - (self.n_windows * self.test_horizon)
        
        for i in range(self.n_windows):
            cutoff = start_test_idx + (i * self.test_horizon)
            end_test = cutoff + self.test_horizon
            
            # --- LÓGICA SLIDING WINDOW ---
            # El inicio del entrenamiento se mueve, manteniendo el tamaño fijo
            start_train = cutoff - self.input_window_size
            
            # 1. Train Set: Tamaño Fijo
            train = df.iloc[start_train:cutoff].copy()
            
            # 2. Test Set: Horizonte Futuro (Normalmente 1)
            test = df.iloc[cutoff:end_test].copy()
            
            yield train, test