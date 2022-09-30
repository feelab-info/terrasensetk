from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor as GBR
from ...performance.metrics import RegressionMetrics
class GradientBoostingRegressor(IAlgorithm):
    """Implementation of the GradientBoostingRegressor from the scikitlearn library
    """
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
        return GradientBoostingRegressor(self.get_params())

    def get_params(self):
        return self.model.get_params()

    def set_params(self,params):
        return self.model.set_params(**params)
        
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        
        loss = trial.suggest_categorical('loss',['squared_error', 'absolute_error', 'huber', 'quantile'])
        learning_rate = trial.suggest_float('learning_rate',0.05,0.5,step=0.05)
        n_estimators = trial.suggest_int('n_estimators', 200, 1000,step=100)
        
        regr= GradientBoostingRegressor({'loss' : loss, 'learning_rate' : learning_rate,'n_estimators' : n_estimators})
        
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)