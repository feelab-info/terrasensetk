from typing import Type
from algorithms.IAlgorithm import IAlgorithm
from algorithms.IFeatureSelection import IFeatureSelection
from dataset.Dataset import Dataset

class Experiment:
    def __init__(self, dataset, feature_selection=None, model=None):
        if(not issubclass(Dataset,dataset)): raise TypeError("Not a subtype of Dataset")
        if(not issubclass(IFeatureSelection,feature_selection)): raise TypeError("Not a subtype of IFeatureSelection")
        if(not issubclass(IAlgorithm,model)): raise TypeError("Not a subtype of IAlgorithm")

        self.dataset = dataset
        self.feature_selection = feature_selection
        self.model = model

    def perform_feature_selection(model,Parser):

    ####
    #impl
     modelo = SelectKBest(args)
    dataset = Dataset()
     Learning().perform_feature_selection(modelo,Parser(dataset))