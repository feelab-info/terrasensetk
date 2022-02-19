
from performance.metrics.MetricsBase import MetricsBase
from utils.Results import Results
from ..algorithms.IAlgorithm import IAlgorithm
from ..algorithms.FeatureSelection.IFeatureSelection import IFeatureSelection
from ..performance.ICrossValidation import ICrossValidation
from ..dataset.Dataset import Dataset
from ..parser.Parser import Parser
from sklearn.model_selection import train_test_split
class Experiment:
    def __init__(self, dataset_parser,model, feature_selection=None, cross_validation=None,fit_for_variable="N",train_test_split=60):
        if(not issubclass(Parser,dataset_parser)): raise TypeError("Not a subtype of Parser")
        if(not issubclass(IFeatureSelection,feature_selection) or not feature_selection is None): raise TypeError("Not a subtype of IFeatureSelection")
        if(not issubclass(IAlgorithm,model)): raise TypeError("Not a subtype of IAlgorithm")
        if(not issubclass(ICrossValidation,model) or not cross_validation is None): raise TypeError("Not a subtype of ICrossValidation")

        self.dataset_parser = dataset_parser
        self.feature_selection = feature_selection
        self.cross_validation = cross_validation
        self.model = model
        self.eopatch_ids = self.dataset_parser.create_dataframe(self.dataset_parser.dataset)["EOPATCH"]
        self.eopatch_ids = self.eopatch_ids.unique()
        self.dataset_array = self.dataset_parser.create_array(fit_for_variable);
        self.fit_for_variable = fit_for_variable
        self.train_test_split=train_test_split


    def execute(self) -> list(Results):
        x,y = self.dataset_array
        features=None
        if(self.feature_selection is not None):
            x = self.feature_selection.fit(self.dataset_array)
            features = self.feature_selection.get_model().get_feature_names_out(self.dataset_parser.features)
        results = []
        if(self.cross_validation is not None):
            folds = self.cross_validation.split(self.eopatch_ids)

            for i,(train,test) in folds:
                model = self.model.clone(self.model)
                x_train,y_train = self.dataset_parser.create_array(self.fit_for_variable,image_ids=train,features=features)
                x_test,y_test = self.dataset_parser.create_array(self.fit_for_variable,image_ids=test,features=features)
                model.fit(x_train,y_train)
                model.predict(x_test)
                results.append(Results(x_train,y_test,x_train,y_train,model))
        else:
            train,test = train_test_split(self.eopatch_ids)
            model = self.model.clone(self.model)
            x_train,y_train = self.dataset_parser.create_array(self.fit_for_variable,image_ids=train,features=features)
            x_test,y_test = self.dataset_parser.create_array(self.fit_for_variable,image_ids=test,features=features)
            model.fit(x_train,y_train)
            model.predict(x_test)
            results.append(Results(x_train,y_test,x_train,y_train,model))
        self.results = results
        return self.results



    def calculate_metrics(self,metrics,list_of_metrics=['rmse']):
        if(self.results is None): raise TypeError("Execute method was not called yet.")
        if not issubclass(metrics,MetricsBase):
            raise TypeError("Metrics is not a subtype of MetricsBase")
    
        return metrics.check_metrics(self.results,list_of_metrics)


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