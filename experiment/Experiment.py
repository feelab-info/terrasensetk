from algorithms.IAlgorithm import IAlgorithm
from performance.IFeatureSelection import IFeatureSelection
from algorithms.ICrossValidation import ICrossValidation
from dataset.Dataset import Dataset
from parser.Parser import Parser

class Experiment:
    def __init__(self, dataset_parser,model, feature_selection=None, cross_validation=None,fit_for_variable="N"):
        if(not issubclass(Parser,dataset_parser)): raise TypeError("Not a subtype of Parser")
        if(not issubclass(IFeatureSelection,feature_selection) or not feature_selection is None): raise TypeError("Not a subtype of IFeatureSelection")
        if(not issubclass(IAlgorithm,model)): raise TypeError("Not a subtype of IAlgorithm")
        if(not issubclass(ICrossValidation,model) or not cross_validation is None): raise TypeError("Not a subtype of ICrossValidation")

        self.dataset_parser = dataset_parser
        self.feature_selection = feature_selection
        self.cross_validation = cross_validation
        self.model = model
        self.dataset_parser.create_dataframe(self.dataset_parser.dataset)
        self.dataset_array = self.dataset_parser.create_array(fit_for_variable);


    def execute(self):
        x,y = self.dataset_array
        
        if(self.feature_selection is not None):
            x = self.feature_selection.fit(self.dataset_array)

        if(self.cross_validation is not None):
            folds = self.cross_validation.split()



        #TODO: prepare the data check
        #TODO: feed the featureSelector check
        #perform kfold if exists
        #create Experiment object with information regarding the folds values and everything that we got
        #do metrics evaluation
        self.dataset_parser

    
    #impl
    """ modelo = SelectKBest(args)
    dataset = Dataset()
     Learning().perform_feature_selection(modelo,Parser(dataset))"""

"""
    def perform_feature_selection(model):
        self.feature_selection.fit()
        pass
    def perform_cross_validation(self,cv):
        if(not issubclass(ICrossValidation,cv)): raise TypeError("Not a subtype of ICrossValidation")
        return cv.split(self.dataset_parser.create_dataframe(self.dataset_parser.dataset))
"""