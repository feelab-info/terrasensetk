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
        self.subset = subset
        self.dataset = dataset
    
    def get_dataset(self):
        return self.dataset

    dataset = property(get_dataset,None,None,"""The dataset that was parsed""")

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
            indices = INDICES+["EOPATCHES","N","P","K"]
        if bands == None:
            bands = BANDS
        self.indices = indices
        self.bands = bands
        if(self.save_dataframe == None):
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
        
    def create_array(self, variable_to_fit):
        """Creates an array ready for fitting (x,y)

        Args:
            variable_to_fit (string): Which variable from create_dataframe it is supposed to use for the y values

        Raises:
            ExecError: [description]
        """
        if(self.save_dataframe == None):
            raise ExecError("Parser.create_dataframe must be called before")
        x_values = self.save_dataframe.round(2)[self.indices+self.bands].values.reshape(-1,len(self.indices+self.bands))
        y_values = self.save_dataframe.round(2)[variable_to_fit].values.reshape(-1,1)        
        return (x_values,y_values)
