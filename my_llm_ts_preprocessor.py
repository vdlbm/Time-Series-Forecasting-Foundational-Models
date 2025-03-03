import numpy as np

class TimeSeriesPreprocessor:
    def __init__(self, percentile=99):
        """
        Inicializa el preprocesador con un percentil para el reescalado.
        """
        self.percentile = percentile
        self.scale_factor = None

    def convert_numbers_to_strings(self, series):
        """
        Convierte una serie numérica en una representación string adecuada para LLMs.
        - Asegura siempre dos decimales con .format()
        - Separa los dígitos con espacios.
        - Usa comas para separar los valores en la secuencia.
        """
        formatted_series = []
        for num in series:
            # Redondear y asegurar dos decimales siempre
            num_str = "{:.2f}".format(num).replace('.', '')  
            # Separar cada dígito con un espacio
            spaced_str = ' '.join(num_str)
            formatted_series.append(spaced_str)

        return ', '.join(formatted_series)  # Separar valores con coma

    def tokenize_numbers(self, formatted_series):
        """
        Tokeniza la serie asegurando la separación individual de dígitos.
        """
        return formatted_series.split(', ')

    def rescale_series(self, series):
        """
        Reescala los valores para que un percentil específico se ajuste a 1.
        """
        if self.scale_factor is None:
            self.scale_factor = np.percentile(series, self.percentile)

        return [round(num / self.scale_factor, 2) for num in series]

    def preprocess(self, series):
        """
        Ejecuta todo el pipeline: reescalado, conversión y tokenización.
        """
        rescaled_series = self.rescale_series(series)
        formatted_series = self.convert_numbers_to_strings(rescaled_series)
        tokenized_series = self.tokenize_numbers(formatted_series)

        return tokenized_series
