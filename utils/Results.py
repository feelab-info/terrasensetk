
class Results:

    def __init__(self,x_test,y_test,x_train,y_train,model,features,parameters,ids_train,ids_test,study=None):
        self._x_test = x_test
        self._y_test = y_test
        self._x_train = x_train
        self._y_train = y_train
        self._model = model
        self._features = features
        self._parameters = parameters
        self._study = study
        self._ids_train = ids_train
        self._ids_test = ids_test
    
    def _get_x_test(self):
        return self._x_test
    def _get_y_test(self):
        return self._y_test
    def _get_x_train(self):
        return self._x_train
    def _get_y_train(self):
        return self._y_train
    def _get_model(self):
        return self._model
    def _get_features(self):
        return self._features
    def _get_parameters(self):
        return self._parameters
    def _get_y_pred(self):
        return self.model.predict(self.x_test)
    def _get_study(self):
        return self._study
    def _get_ids_test(self):
        return self._ids_test
    def _get_ids_train(self):
        return self._ids_train

    def __str__(self):
        return f"Results Object: [\n    x_test: {type(self._get_x_test())}\n    y_test: {type(self._get_y_test())}\n    x_train: {type(self._get_x_train())}\n    y_train: {type(self._get_y_train())}\n    model: {self._get_model()}\n    parameters: {self._get_parameters()}\n    features: {self._get_features()}\n    y_pred: {type(self._get_y_pred())}\n    study: {type(self._get_study())}\n    ids_test: {len(self._get_ids_test())}\n    ids_train: {len(self._get_ids_train())}\n]"

    def __repr__(self):
        return self.__str__()

    x_test = property(_get_x_test,None,None,"Gets the x_test")
    y_test = property(_get_y_test,None,None,"Gets the y_test aka groundtruth")
    x_train = property(_get_x_train,None,None,"Gets the x_train")
    y_train = property(_get_y_train,None,None,"Gets the y_train")
    model = property(_get_model,None,None,"Gets the model")
    parameters = property(_get_parameters,None,None,"Get the models parameters")
    features = property(_get_features,None,None,"Gets the features")
    y_pred = property(_get_y_pred,None,None,"Get the corresponding prediction for x_test")
    study = property(_get_study,None,None,"The Hyperoptimization study")
    ids_train = property(_get_ids_train,None,None,"The Range of indices for the train eopatches")
    ids_test = property(_get_ids_test,None,None,"The Range of indices for the test eopatches")
    
