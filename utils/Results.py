
class Results:

    def __init__(self,x_test,y_test,x_train,y_train,model,features):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.features = features
    
    def _get_x_test(self):
        return self.x_test
    def _get_y_test(self):
        return self.y_test
    def _get_x_train(self):
        return self.x_train
    def _get_y_train(self):
        return self.y_train
    def _get_model(self):
        return self.model
    def _get_features(self):
        return self.features
    def _get_y_pred(self):
        return self.model.predict(self.x_test)
        
    x_test = property(_get_x_test,None,None,"Gets the x_test")
    y_test = property(_get_y_test,None,None,"Gets the y_test")
    x_train = property(_get_x_train,None,None,"Gets the x_train")
    y_train = property(_get_y_train,None,None,"Gets the y_train")
    model = property(_get_model,None,None,"Gets the model")
    features = property(_get_features,None,None,"Gets the features")
    y_pred = property(_get_y_pred,None,None,"Get the corresponding prediction for x_test")
    
