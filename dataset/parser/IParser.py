
import abc
class IParser(abc.ABC):
    
    @abc.abstractmethod
    def convert(self, variable_to_fit,image_ids=None,features=None):
        raise NotImplementedError

    def create_dataframe():
        pass