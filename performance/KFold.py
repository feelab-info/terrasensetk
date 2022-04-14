from .ICrossValidation import ICrossValidation
from sklearn.model_selection import KFold as sklKFold
class KFold(ICrossValidation):

    def __init__(self,*args):
        self.model = sklKFold(*args)

    def split(self,values_to_split):
        return self.model.split(values_to_split)

