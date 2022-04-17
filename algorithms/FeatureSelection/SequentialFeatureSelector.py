from numpy import number
from sklearn import clone
from .IFeatureSelection import IFeatureSelection
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import SequentialFeatureSelector as sklearnFeatureSelector

from sklearn.ensemble import GradientBoostingRegressor
class SequentialFeatureSelector(IFeatureSelection):

    def __init__(self,number_of_features=5,estimator=GradientBoostingRegressor(),*args):
        self.model = sklearnFeatureSelector(estimator=estimator,n_features_to_select=number_of_features,*args)
        self.number_of_features = number_of_features

    def fit(self,x,y,*args):
        return self.model.fit_transform(x,y,*args)

    def predict(self,*kwargs):
        return self.model.predict(*kwargs)

    def get_model(self):
        return self.model

    def clone(self):
        return clone(self.model)

    def get_params(self):
        return self.model.get_params()