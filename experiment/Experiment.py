
from ..performance.metrics.IMetrics import IMetrics
from ..utils.Results import Results
from ..algorithms.IAlgorithm import IAlgorithm
from ..algorithms.FeatureSelection.IFeatureSelection import IFeatureSelection
from ..performance.ICrossValidation import ICrossValidation
from ..dataset import Dataset
from ..dataset.parser import IParser
import optuna
from sklearn.model_selection import train_test_split
class Experiment:
    def __init__(self,name, dataset_parser,models, feature_selection=None, cross_validation=None,input_features=None,fit_for_variable="N",train_test_split=0.60):
        
        self._check_args(name, dataset_parser, models, feature_selection, cross_validation, train_test_split)

        self.name = name
        self.dataset_parser = dataset_parser
        self.feature_selection = feature_selection
        self.cross_validation = cross_validation
        self.models = models
        self.eopatch_ids = self.dataset_parser.create_dataframe(image_identifier="Point_ID")["Point_ID"]
        self.eopatch_ids = self.eopatch_ids.unique()
        self.dataset_array = self.dataset_parser.convert(fit_for_variable)
        self.fit_for_variable = fit_for_variable
        self.train_test_split=train_test_split
        self.input_features = input_features
        self.x,self.y,_ = self.dataset_array
        self.feature_selection_complete=False
        self.cross_validation_complete=False
        self.folds=None
        self.study=None

    def execute_feature_selection(self,feature_selection=None):

        if(self.feature_selection is not None):
            self.feature_selection.fit(self.x,self.y)
        elif(issubclass(type(feature_selection),IFeatureSelection) or feature_selection is not None):
            self.feature_selection.fit(self.x,self.y)
        else:
            raise Exception("No feature selection provided!")
        self.feature_selection_complete=True
        self.features = self.feature_selection.get_model().get_feature_names_out(self.dataset_parser.features)

        return self.features

    def execute_cross_validation(self):
        if self.folds: 
            return self.folds
        if self.cross_validation is not None:
            self.folds = list(self.cross_validation.split(self.eopatch_ids))
            self.cross_validation_complete=True
            return self.folds
        raise Exception("No cross validation provided!")

    def execute(self, perform_optimization=True,n_trials=100,folds=None):
        
        rand_state = 1337
        self.y_interval = self.y.max()-self.y.min()
        features = self._define_features()
        self.x,self.y,_ = self.dataset_parser.convert(self.fit_for_variable,features=features)
        x_train,x_test = train_test_split(self.x,random_state = rand_state,train_size=self.train_test_split)
        y_train,y_test = train_test_split(self.y,random_state = rand_state,train_size=self.train_test_split)
        if perform_optimization:
            for i,model in enumerate(self.models):
                self.study=None
                try:
                    self.study = optuna.create_study(direction='minimize')
                    self.study.optimize(lambda trial: model.objective_function(trial,x_train,y_train,x_test,y_test),n_trials=n_trials)
                    params = self.study.best_params
                    model.set_params(params)

                except NotImplementedError:
                    print(f"HPO: The model {model} doesn't contain an objective function, the model will be trained with the supplied arguments")
                    
        results = {}
        if(self.cross_validation is not None or folds is not None):
            if folds is not None:
                self.folds = folds
            elif self.folds is None:
                self.execute_cross_validation()
                
            for j,model in enumerate(self.models):
                    results[j] = []
                    for i,(train,test) in enumerate(self.folds):
                        results[j].append(self._get_results_for_model(features, model, self.eopatch_ids[train], self.eopatch_ids[test]))
            
        else:
            train,test = train_test_split(self.eopatch_ids,train_size=self.train_test_split)
            for i,model in enumerate(self.models):
                results[i] = []
                results[i].append(self._get_results_for_model(features, model, train, test))
        self.results = results
        return self.results

    def _define_features(self):
        features=self.dataset_parser.features
        
        if(not self.feature_selection_complete):
            if(self.feature_selection is not None):
               features = self.execute_feature_selection()
            elif(self.input_features is not None):
                for feature in self.input_features:
                    if(feature not in features): raise TypeError(f"The feature '{feature}' is not in {features}")
                features = list(set(features) & set(self.input_features))
        else:
            features = self.features
        return features

    def _get_results_for_model(self, features, model, train, test):
        model = model.clone()
        x_train,y_train,ids_train = self.dataset_parser.convert(self.fit_for_variable,image_ids=train,features=features)
        x_test,y_test,ids_test = self.dataset_parser.convert(self.fit_for_variable,image_ids=test,features=features)
        model.fit(x_train,y_train)
        model.predict(x_test)
        return Results(x_test,y_test,x_train,y_train,model,features,model.get_params(),ids_train,ids_test,self.study)

    def calculate_metrics(self,metrics,list_of_metrics=['rmse','mae']):
        if(self.results is None): raise TypeError("Execute method was not called yet.")
        if not issubclass(type(metrics),IMetrics):
            raise TypeError("Metrics is not a subtype of MetricsBase")
        self.metrics_results = metrics.check_metrics(self.results,list_of_metrics,self.y_interval)
        return self.metrics_results

    def log_runs(self):
        import mlflow
        run_name = self.name
        for j,model_batch in self.results.items():
            for i, run in enumerate(model_batch):
                with mlflow.start_run(run_name=f"{run.model.get_name()} - run {i}"):
                    for k,v in run.model.get_params().items():
                        mlflow.log_param(k,v)
                    metrics = self.metrics_results.columns[2:]
                    for metric in metrics:
                        mlflow.log_metric(metric,self.metrics_results.query("algorithm == @run.model.get_name() and run == @i+1")[metric].values[0])
                    #mlflow.log_artifacts(run)
        # with mlflow.start_run(run_name=run_name):
        #       mlflow.log_param("batch_size", batch_size)
        #       mlflow.log_param("learning_rate", learning_rate)
        #       mlflow.log_param("epochs", epochs)
        #       mlflow.log_metric("train_loss", train_loss)
        #       mlflow.log_metric("train_accuracy", train_acc)
        #       mlflow.log_metric("val_loss", val_loss)
        #       mlflow.log_metric("val_accuracy", val_acc)
        #       mlflow.log_artifacts(self.results)

    def _check_args(self, name, dataset_parser, models, feature_selection, cross_validation, train_test_split):
        if(not issubclass(type(dataset_parser),IParser)): raise TypeError(f"{type(dataset_parser)} is not a subtype of IParser")
        if(not issubclass(type(feature_selection),IFeatureSelection) and feature_selection is not None): raise TypeError(f"{type(feature_selection)} is not a subtype of IFeatureSelection")
        for i in models:
            if(not issubclass(type(i),IAlgorithm)): raise TypeError(f"{type(i)} is not a subtype of IAlgorithm")    
        if(not issubclass(type(cross_validation),ICrossValidation) and cross_validation is not None): raise TypeError(f"{type(cross_validation)} is not a subtype of ICrossValidation")
        if(not isinstance(name,str)): raise TypeError("An Experiment name must be provided")
        if(train_test_split>=1 or train_test_split<=0): raise TypeError("The train_test_split value should be a float between 0 and 1")
