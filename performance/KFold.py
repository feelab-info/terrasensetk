from .ICrossValidation import ICrossValidation
from sklearn.model_selection import KFold as sklKFold
class KFold(ICrossValidation):
    """Implementation of the KFold from the scikitlearn library
    """
    def __init__(self,args={},**kwargs):
        model = sklKFold
        if args:
            self.model = model(**args)
        elif kwargs:
            self.model = model(**kwargs)
        else:
            self.model = model()

    def split(self,values_to_split):
        return self.model.split(values_to_split)

