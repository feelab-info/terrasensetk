from numpy import number
from IFeatureSelection import IFeatureSelection
from sklearn.feature_selection import SelectKBest,f_regression, SequentialFeatureSelector

class SequentialFeatureSelector(IFeatureSelection):

    def __init__(self,number_of_features,*args):
        self.model = SequentialFeatureSelector(_features_to_select=number_of_features,*args)
        self.number_of_features = number_of_features

    def fit(self,*kwargs):
        return self.model.fit(*kwargs)

    def predict(self,*kwargs):
        return self.model.predict(*kwargs)

    def get_model(self):
        return self.model