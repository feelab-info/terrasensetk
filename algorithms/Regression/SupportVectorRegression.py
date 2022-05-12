from sklearn.svm import SVR
from ..IAlgorithm import IAlgorithm
from sklearn.base import clone as skclone
from ...performance.metrics import RegressionMetrics 
class SupportVectorRegression(IAlgorithm):

    def __init__(self,kargs=None):
        if kargs is None:
            self.model = SVR()
            return
        self.model = SVR(**kargs)

    def fit(self,x_values,y_values,*args):
        return self.model.fit(x_values,y_values,*args)

    def predict(self,x_values):
        if(self.model.fit_status_ != 0): 
            raise TypeError("The model is not fitted yet")
        return self.model.predict(x_values)

    def get_model(self):
        return self.model
    
    def clone(self):
        return skclone(self.model)

    def get_params(self):
        return self.model.get_params()

    def set_params(self,params):
        return self.model.set_params(**params)

    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        kernel=trial.suggest_categorical('kernel',['rbf','poly','linear','sigmoid'])
        c=trial.suggest_float("C",0.1,3.0)
        gamma=trial.suggest_categorical('gamma',['auto','scale'])
        degree=trial.suggest_int("degree",1,3)

        regr = SupportVectorRegression({'kernel': kernel, 'C': c, 'gamma': gamma, 'degree': degree,'n_jobs':2})
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)
