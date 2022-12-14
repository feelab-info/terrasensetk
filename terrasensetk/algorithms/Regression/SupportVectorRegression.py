from sklearn.svm import SVR
from ..IAlgorithm import IAlgorithm
from sklearn.base import clone as skclone
from ...performance.metrics import RegressionMetrics 
class SupportVectorRegression(IAlgorithm):
    """Implementation of the SupportVectorRegression from the scikitlearn library
    """
    def __init__(self,args={},**kwargs):
        model = SVR
        if args:
            self.model = model(**args)
        elif kwargs:
            self.model = model(**kwargs)
        else:
            self.model = model()

    def fit(self,x_values,y_values,*args):
        return self.model.fit(x_values,y_values,*args)

    def predict(self,x_values):
        if(self.model.fit_status_ != 0): 
            raise TypeError("The model is not fitted yet")
        return self.model.predict(x_values)

    def get_model(self):
        return self.model
    
    def clone(self):
        return SupportVectorRegression(self.get_params())

    def get_params(self):
        return self.model.get_params()

    def set_params(self,params):
        return self.model.set_params(**params)

    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        metric = RegressionMetrics()
        kernel=trial.suggest_categorical('kernel',['rbf','sigmoid'])
        c=trial.suggest_float("C",0.1,3.0,step=0.5)
        gamma=trial.suggest_categorical('gamma',['auto','scale'])
        #degree=trial.suggest_int("degree",1,3)
        regr = SupportVectorRegression({'kernel': kernel, 'C': c, 'gamma': gamma})
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        return metric.cmd_rmse(y_test, y_pred)
