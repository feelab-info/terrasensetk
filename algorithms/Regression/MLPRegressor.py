from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor as MLPR
from ...performance.metrics import RegressionMetrics

class MLPRegressor(IAlgorithm):

    def __init__(self,args={},**kwargs):
        model = MLPR
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
        return MLPRegressor(self.get_params())

    def get_params(self):
        return self.model.get_params()

    def set_params(self,params):
        return self.model.set_params(**params)
        
    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        random_state = trial.suggest_int('random_state', 1, 300)
        max_iter =  trial.suggest_int('max_iter', 200, 500)
        activation = trial.suggest_categorical('activation',['identity', 'logistic', 'tanh', 'relu'])
        learning_rate = trial.suggest_categorical('learning_rate',['constant', 'invscaling', 'adaptive'])
        solver = trial.suggest_categorical('solver', ['adam','sgd','lbfgs'])
        if solver in ['adam','sgd']:
            learning_rate_init= trial.suggest_float('learning_rate_init',0.0001,1.5) 
            regr = MLPRegressor({'learning_rate': learning_rate, 'max_iter': max_iter, 'random_state': random_state,'activation': activation,'learning_rate_init': learning_rate_init})
        else:
            regr= MLPRegressor({'learning_rate' : learning_rate, 'max_iter' : max_iter,'random_state' : random_state,'activation' : activation})
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)