import abc
class IAlgorithm(abc.ABC):


    @abc.abstractmethod
    def fit(x,y,**kwargs):
        pass

    @abc.abstractmethod
    def predict(x,**kwargs):
        pass

    @abc.abstractmethod
    def clone(self):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass
    
    def set_params(self,params):
        raise NotImplementedError(f"set_params for {type(self)} not implemented")

    def objective_function(self,trial,x_train,y_train,x_test,y_test):
        raise NotImplementedError(f"Objective function for {type(self)} not implemented")