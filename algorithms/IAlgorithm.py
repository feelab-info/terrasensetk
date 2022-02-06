import abc
class IAlgorithm(abc.ABC):


    @abc.abstractmethod
    def fit(*kwargs):
        pass


    @abc.abstractmethod
    def predict(*kwargs):
        pass

    @abc.abstractclassmethod
    def clone(cls):
        pass