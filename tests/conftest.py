import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False
@pytest.fixture(scope="session")
def sample_data():
    """
    Fixture: synthetic daily time series (50 points) with trend + seasonality.
    Columns: ['ds', 'y']. Used across multiple test modules.
    """
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    values = [100 + i + (10 * (i % 7)) for i in range(50)]
    
    df = pd.DataFrame({'ds': dates, 'y': values})
    return df

@pytest.fixture(scope="session")
def sample_data_business_days():
    """
    Fixture: synthetic business-day series (200 points) with gaps on weekends.
    Simulates SP500/EURUSD trading calendar.
    """
    dates = pd.bdate_range(start='2022-01-03', periods=200)
    values = [4500 + i * 2 + np.sin(i / 5) * 50 for i in range(200)]
    return pd.DataFrame({'ds': dates, 'y': values})

@pytest.fixture(scope="session")
def sample_data_with_gaps():
    """
    Fixture: daily series with intentional gaps (simulates holidays).
    Has a gap on 2023-01-16 (MLK Day) and 2023-02-20 (Presidents Day).
    """
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    values = [100 + i + np.random.normal(0, 2) for i in range(90)]
    df = pd.DataFrame({'ds': dates, 'y': values})
    # Remove some dates to simulate holidays
    holidays = pd.to_datetime(['2023-01-16', '2023-02-20'])
    df = df[~df['ds'].isin(holidays)].reset_index(drop=True)
    return df

@pytest.fixture
def gpu_available():
    """Returns True if GPU available for heavy tests."""
    return _torch_available and torch.cuda.is_available()
