from numpy import number
from .IFeatureSelection import IFeatureSelection
from sklearn.feature_selection import SelectKBest,f_regression, SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor
class SequentialFeatureSelector(IFeatureSelection):

    def __init__(self,*args,number_of_features=5,estimator=GradientBoostingRegressor()):
        self.model = SequentialFeatureSelector(estimator=estimator,n_features_to_select=number_of_features,*args)
        self.number_of_features = number_of_features

    def fit(self,*kwargs):
        return self.model.fit_transform(*kwargs[0],*kwargs[1])

    def predict(self,*kwargs):
        return self.model.predict(*kwargs)

    def get_model(self):
        return self.model