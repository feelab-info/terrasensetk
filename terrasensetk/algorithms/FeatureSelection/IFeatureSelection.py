import abc

from numpy import number

from ..IAlgorithm import IAlgorithm
class IFeatureSelection(IAlgorithm):
    """Base class for implementation of the Feature Selection Algorithms
    """

    def __init__(self,number_of_features,*kwargs):
        self.number_of_features = number_of_features

    def set_number_of_features(self,number_of_features):
        self.number_of_features = number_of_features
        
    def get_number_of_features(self) -> int:
        return self.number_of_features