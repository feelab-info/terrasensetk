from sqlalchemy import true
from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor as RFR
from ...performance.metrics import RegressionMetrics

class RandomForestRegressor(IAlgorithm):
    """Implementation of the RandomForestRegressor from the scikitlearn library
    """
    def __init__(self,args={},**kwargs):
        model = RFR
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
        return RandomForestRegressor(self.get_params())

    def get_params(self):
        return self.model.get_params()
    
    def set_params(self,params):
        return self.model.set_params(**params)
    
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
        # bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
        # max_depth = trial.suggest_int('max_depth', 1, 200)
        max_features = trial.suggest_categorical('max_features', ['sqrt','log2'])
        # max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 2000)
        n_estimators = trial.suggest_int('n_estimators', 30, 300)
        regr = RandomForestRegressor({'bootstrap': True, 'criterion': criterion,
                                    'max_features': max_features,
                                    'n_estimators': n_estimators,'n_jobs':2})
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)
