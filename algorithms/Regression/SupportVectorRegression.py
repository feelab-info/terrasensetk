from sklearn.svm import SVR
from ..IAlgorithm import IAlgorithm
from sklearn.base import clone
class SupportVectorRegression(IAlgorithm):

    def __init__(self,*args,**kargs):
        self.model = SVR(*args)

    def fit(self,x_values,y_values,*args):
        return self.model.fit(x_values,y_values,*args)

    def predict(self,x_values):
        if(not self.model.is_fit()): 
            raise TypeError("The model is not fitted yet")
        return self.model.predict(x_values)

    def get_model(self):
        return self.model
    
    def clone(self):
        return clone(self.model)

    def get_params(self):
        return self.model.get_params()
