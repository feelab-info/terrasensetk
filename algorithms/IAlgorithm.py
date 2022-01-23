import abc
class IAlgorithm(abc.ABC):


    @abc.abstractclassmethod
    def fit(*kwargs):
        pass


    @abc.abstractclassmethod
    def predict(*kwargs):
        pass