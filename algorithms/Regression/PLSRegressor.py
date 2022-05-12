from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression as PLSR

class PLSRegressor(IAlgorithm):

    def __init__(self,kargs=None):
        if kargs is None:
            self.model = PLSR()
            return
        self.model = PLSR(**kargs)

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

    def set_params(self,params):
        return self.model.set_params(**params)
        
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        raise NotImplementedError("Nop")