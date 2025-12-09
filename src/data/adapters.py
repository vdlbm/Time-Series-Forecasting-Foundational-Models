import pandas as pd
import re
from typing import List

class DataAdapter:
    """
    Static utility class to translate between Numerical DataFrames and 
    LLM-friendly Text formats.
    
    Implements a 'Semantic Encoding' strategy to leverage the LLM's 
    natural language processing capabilities for time series forecasting.
    """
    
    @staticmethod
    def to_llm_prompt(df: pd.DataFrame, window_size: int = 90) -> str:
        """
        Converts the tail of a time series into a narrative text sequence.
        
        Strategy:
        Instead of raw numbers ("100, 102"), we use semantic descriptors 
        ("from 100 increasing to 102"). This helps the LLM understand 
        the derivative (direction and magnitude) of the change.
        
        Args:
            df (pd.DataFrame): Input data containing 'y' column.
            window_size (int): Number of historical points to include.
            
        Returns:
            str: A comma-separated string describing the trends.
        """
        # 1. Slice Data to Context Window
        # We take the last N points defined by window_size
        subset = df.tail(window_size + 1).copy()
        series = subset['y'].values
        text_parts = []
        
        # 2. Semantic Encoding Loop
        for i in range(len(series) - 1):
            val_curr = series[i]
            val_next = series[i+1]
            
            # Skip if we encounter any residual NaNs
            if pd.isna(val_curr) or pd.isna(val_next):
                continue
                
            # Describe the transition
            if val_next > val_curr:
                text_parts.append(f"from {val_curr:.2f} increasing to {val_next:.2f}")
            elif val_next < val_curr:
                text_parts.append(f"from {val_curr:.2f} decreasing to {val_next:.2f}")
            else:
                text_parts.append(f"remains flat at {val_curr:.2f}")
        
        return ", ".join(text_parts)

    @staticmethod
    def parse_llm_output(text: str) -> List[float]:
        """
        Extracts numerical values from the LLM's text response.
        
        Why this is needed:
        LLMs are chatty. Even with strict prompting, they might output:
        "Based on the pattern, the next value is 105.2".
        We need to robustly extract '105.2' and ignore the text.
        
        Args:
            text (str): The raw string output from the LLM.
            
        Returns:
            List[float]: A list of extracted numerical values.
        """
        # Regex explanation:
        # -?      : Optional negative sign
        # \d+     : One or more digits
        # (?:     : Non-capturing group for decimals
        # \.\d+   : A dot followed by digits
        # )?      : The decimal part is optional
        matches = re.findall(r'-?\d+(?:\.\d+)?', text)
        
        if not matches:
            return []
            
        return [float(x) for x in matches]