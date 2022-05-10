
import abc
class IParser(abc.ABC):
    
    @abc.abstractmethod
    def convert(self, variable_to_fit,image_ids=None,features=None):
        """Creates an array ready for fitting (x,y)

        Args:
            variable_to_fit (string): Which variable from create_dataframe it is supposed to use for the y values
            image_ids (list of int): The images in which it is supposed to include in the values, 
                if None given, it will return all the images given in the dataframe
            features (list of string): Name of the columns in which to include in the values,
                if None given, it will return all the image features included in the dataframe
        Raises:
            ExecError: If the create_dataframe method wasn't called before.
        """
        raise NotImplementedError

    def create_dataframe():
        pass