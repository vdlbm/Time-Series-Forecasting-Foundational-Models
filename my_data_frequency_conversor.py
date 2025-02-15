import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import datetime as dt


def mins_to_hours(df):
    """
    Converts data with minute frequency to hourly frequency
    """

    # Separar los datos en columnas individuales
    #df[['DateTime', 'Open', 'Close']] = df['DateTime;Open;Close'].str.split(';', expand=True)

    # Convertir la columna 'DateTime' a tipo de datos de fecha y hora
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Establecer 'DateTime' como índice
    #df.set_index('DateTime', inplace=True)

    # Filtrar los datos para obtener solo los valores en los horarios deseados (por ejemplo, 9:30, 10:30, etc.)
    filtered_df = df[df.index.strftime('%H:%M') == '09:30']

    # Puedes agregar más horarios si lo deseas
    horarios_deseados = ['14:30', '15:30', '16:30', '17:30', '18:30', '19:30', '20:30', '21:30', '22:30', '23:30']
    filtered_df = df[df.index.strftime('%H:%M').isin(horarios_deseados)]
    
    return filtered_df