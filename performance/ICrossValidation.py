
import abc

class ICrossValidation(abc.ABC):
    """Base class to implement cross validation algorithms
    """
    @abc.abstractclassmethod
    def split(self,values_to_split):
        pass