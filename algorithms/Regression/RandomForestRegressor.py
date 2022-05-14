from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor as RFR
from ...performance.metrics import RegressionMetrics

class RandomForestRegressor(IAlgorithm):

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
        return clone(self.model)

    def get_params(self):
        return self.model.get_params()
    
    def set_params(self,params):
        return self.model.set_params(**params)
    
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        criterion = trial.suggest_categorical('criterion', ['mse', 'mae'])
        bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
        max_depth = trial.suggest_int('max_depth', 1, 10000)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2'])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 10000)
        n_estimators =  trial.suggest_int('n_estimators', 30, 130)
        regr = RandomForestRegressor({'bootstrap': bootstrap, 'criterion': criterion,
                                    'max_depth': max_depth, 'max_features': max_features,
                                    'max_leaf_nodes': max_leaf_nodes,'n_estimators': n_estimators})
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)
