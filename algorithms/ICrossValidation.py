
import abc


class ICrossValidation(abc.ABC):

    @abc.abstractclassmethod
    def split(self,values_to_split):
        pass