from IFeatureSelection import IFeatureSelection
from sklearn.model_selection import KFold as sklKFold
class KFold(IFeatureSelection):

    def __init__(self,*args):
        self.model = sklKFold(*args)

    def split(self,values_to_split):
        self.model.split(values_to_split)

