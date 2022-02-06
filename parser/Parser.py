from shutil import ExecError
import pandas as pd
from typing import Type
from ..dataset import Dataset
INDICES = ["RVI","NDVI74","IRECI","GM","GNDVI","RECHI","REPLI","S2REP","SRRE","NDVI_NB"]
BANDS = ["1","2","3","4","5","6","7","8", "8-A","9","10","11","12"]
class Parser:
    
    def __init__(self,dataset,subset):
        if( not isinstance(dataset,Dataset)):
            
            raise TypeError("Must be a subtype of terrasensetk.Dataset")
        self.indices = INDICES
        self.bands = BANDS
        self.subset = subset
        self.dataset = dataset
    
    def _get_dataset(self):
            return self.dataset
    def _get_features(self):
        return self.indices+self.bands
        
    dataset = property(_get_dataset,None,None,"""The dataset that was parsed""")
    features = property(_get_features,None,None,"""The features that exist in this array""")
    def create_dataframe(self,subset, indices=None, bands=None):
        """Creates the dataframe ready for the algorithm processing

        Args:
            subset (Dataframe): The filtered part of the Dataset.get_eopatches_dataframe() which are supposed to fit
            indices ([type], optional): [description]. Defaults to None.
            bands ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if indices == None:
            indices = INDICES+["EOPATCH","N","P","K"]
        if bands == None:
            bands = BANDS
        self.indices = indices
        self.bands = bands
        if(self.save_dataframe != None):
            return self.save_dataframe
        self.subset = subset
        all_features = indices+bands
        save_dataframe = pd.DataFrame(columns=indices+bands) 
        self.imagery = indices[:-4]
        
        #for i,eopatch in enumerate(self.dataset.get_eopatches()[eopatches_indices]):
        for i,eopatch in enumerate(subset):
            mask = eopatch.get_masked_region()
            values = eopatch.get_values_of_masked_region(indices,bands)

            values["N"] = eopatch.get_dataset_entry_value("N",is_pixelized=True)
            values["P"] = eopatch.get_dataset_entry_value("P",is_pixelized=True)
            values["K"] = eopatch.get_dataset_entry_value("K",is_pixelized=True)
            values["EOPATCH"] = eopatch.get_dataset_entry_value("Point_ID",is_pixelized=True)
            values = pd.DataFrame(values)
            save_dataframe = save_dataframe.append(values, ignore_index=True)

        self.save_dataframe = save_dataframe
        return self.save_dataframe
        
    def create_array(self, variable_to_fit,image_ids=None,features=None):
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

        if(self.save_dataframe == None):
            raise ExecError("Parser.create_dataframe must be called before")
        if(image_ids == None):
            vals = self.save_dataframe.round(2)            
        else:
            vals = self.save_dataframe.round(2)[self.save_dataframe["EOPATCH"].isin(image_ids)]
        if(features == None):
            x_values = vals[self.indices+self.bands].values.reshape(-1,len(self.indices+self.bands))
        else:
            x_values = vals[features].values.reshape(-1,len(features))
        y_values = vals[variable_to_fit].values.reshape(-1,1).ravel()        
        return (x_values,y_values)
    
    