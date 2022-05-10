
class Results:

    def __init__(self,x_test,y_test,x_train,y_train,model,features,parameters):
        self._x_test = x_test
        self._y_test = y_test
        self._x_train = x_train
        self._y_train = y_train
        self._model = model
        self._features = features
        self._parameters = parameters
    
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
        
    def __str__(self):
        return f"Results Object: [\n    x_test: {type(self._get_x_test())}\n    y_test: {type(self._get_y_test())}\n    x_train: {type(self._get_x_train())}\n    y_train: {type(self._get_y_train())}\n    model: {self._get_model()}\n    parameters: {self._get_parameters()}\n    features: {self._get_features()}\n    y_pred: {type(self._get_y_pred())}]"

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
    
