import numpy as np
import pandas as pd 


class Model(object):
    def __init__(self, name, mse, mae, mape):
        self.name = name
        self.mse = mse
        self.mae = mae
        self.mape = mape

    def getMSE(self):
        return self.mse
    def getMAE(self):
        return self.mae
    def getMAPE(self):
        return self.mape
    
    def setMSE(self, mse):
        self.mse = mse
    def setMAE(self, mae):
        self.mae = mae
    def setMAPE(self, mape):
        self.mape = mape
    