from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor as GBR
from ...performance.metrics import RegressionMetrics
class GradientBoostingRegressor(IAlgorithm):

    def __init__(self,args={},**kwargs):
        model = GBR
        if args:
            self.model = model(**args)
        elif kwargs:
            self.model = model(**kwargs)
        else:
            self.model = model()

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

    def set_params(self,**params):
        return self.model.set_params(**params)
        
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        raise NotImplementedError("GBM HPO not available")