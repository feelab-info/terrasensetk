from ...performance.metrics import RegressionMetrics
from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression as PLSR

class PLSRegressor(IAlgorithm):

    def __init__(self,args={},**kwargs):
        model = PLSR
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
        return PLSRegressor(self.get_params())

    def get_params(self):
        return self.model.get_params()

    def set_params(self,params):
        return self.model.set_params(**params)
        
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        n_components = trial.suggest_int('n_components',1,x_train.shape[1])
        max_iter = trial.suggest_int('max_iter',200,1000,step=50)
        regr = PLSRegressor({'n_components':n_components,'max_iter':max_iter})
        
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)

