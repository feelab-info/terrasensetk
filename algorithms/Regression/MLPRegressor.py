from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor as MLPR
class MLPRegressor(IAlgorithm):

    def __init__(self,*args,**kargs):
        self.model = MLPR(*args)

    def fit(self,x_values,y_values,*args):
        return self.model.fit(x_values,y_values,*args)

    def predict(self,x_values):
        return self.model.predict(x_values)

    def get_model(self):
        return self.model
    
    def clone(self):
        return clone(self.model)

    def get_params(self):
        return self.model.get_params()
