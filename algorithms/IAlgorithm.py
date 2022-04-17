import abc
class IAlgorithm(abc.ABC):


    @abc.abstractmethod
    def fit(x,y,*kwargs):
        pass


    @abc.abstractmethod
    def predict(x,*kwargs):
        pass

    @abc.abstractmethod
    def clone(self):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass