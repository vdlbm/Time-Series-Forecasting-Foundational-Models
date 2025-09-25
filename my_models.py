import numpy as np
import pandas as pd 


class Model(object):
    def __init__(self, name, mse, mae, mape):
        self.name = name
        self.mse = mse
        self.mae = mae
        self.mape = mape
    