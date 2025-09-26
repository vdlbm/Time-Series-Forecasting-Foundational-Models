# my_models.py
class Model(object):
    def __init__(self, name, mse, mae, mape):
        self.name = name
        self.mse = mse
        self.mae = mae
        self.mape = mape

    def __str__(self):
        return f"Model: {self.name}. Metrics--> MSE: {self.mse}, MAE: {self.mae}, MAPE: {self.mape}"