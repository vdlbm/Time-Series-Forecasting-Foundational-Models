import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import datetime as dt


def mins_to_hours(df):
    """
    Converts data with minute frequency to hourly frequency. Since the nyse stock market is open from 9:30 to 16:00, we're selecting
    the data that includes minutes of 30 (9:30, 10:30, 11:30, 12:30, 13:30, 14:30 and 15:30), in our local time (14:30,...).
    """

    # Establecer 'DateTime' como índice
    df.set_index('DateTime', inplace=True)

    # Seleccionar la columna 'Open' y convertirla en una Serie
    s = df['Open']

    # Filtrar la Serie para quedarse solo con los valores donde los minutos son 30
    s_filtered = s[s.index.minute == 30]

    s_filtered = s.filtered.asfreq('h')
    return s_filtered

def mins_to_days(df):
    """
    Converts data with minute frequency to hourly frequency. 
    We´ll be selecting the open price of the stock at 9:30, 14:30 in our local time.
    """

    # Establecer 'DateTime' como índice
    df.set_index('DateTime', inplace=True)

    # Seleccionar la columna 'Open' y convertirla en una Serie
    s = df['Open']

    # Filtrar la Serie para quedarse solo con los valores donde los minutos son 30
    s_filtered = s[(s.index.minute == 30) & (s.index.hour == 14)]

    s_filtered = s.filtered.asfreq('d')

    return s_filtered

def mins_to_months(df):
    """
    Converts data with minute frequency to monthly frequency.
    Selecting the last datapoint of each month.   
    """
    
    # Establecer 'DateTime' como índice
    df.set_index('DateTime', inplace=True)

    # Seleccionar la columna 'Open' y convertirla en una Serie
    s = df['Open']

    # Agrupar los datos por mes y seleccionar el primer dato de cada mes
    s_filtered = s.resample('ME').last()

    s_filtered = s.filtered.asfreq('MS')

    return s_filtered   

def split_data(df):
    """
    Splits the data into train and validation sets.
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    train = df[df.index.year < 2024]
    val = df[df.index.year == 2024]

    return train, val