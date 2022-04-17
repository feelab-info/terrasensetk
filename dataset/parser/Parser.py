from shutil import ExecError
import pandas as pd
from typing import Type
from .. import Dataset
from .IParser import IParser
#INDICES = ["RVI","NDVI74","IRECI","GM","GNDVI","RECHI","REPLI","S2REP","SRRE","NDVI_NB"]
INDICES =["NDVI74","IRECI","REM","GM","REPLI","RECHI","S2REP","NDI45","GNDVI","NDVINB","RVI"],
BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08", "B8A","B09","B10","B11","B12"]
class Parser(IParser):
    
    def __init__(self,dataset,subset = None):
        if( not isinstance(dataset,Dataset)):
            raise TypeError("Must be a subtype of terrasensetk.Dataset")
        if subset is None:
            self.subset = dataset.get_eopatches()
        else:
            self.subset = subset
        self.indices = INDICES
        self.bands = BANDS
        self._dataset = dataset
        self.save_dataframe = None
    
    def _get_dataset(self):
        return self._dataset
    def _get_features(self):
        return self.indices+self.bands
        
    dataset = property(_get_dataset,None,None,"""The dataset that was parsed""")
    features = property(_get_features,None,None,"""The features that exist in this array""")
    
    def create_dataframe(self,subset=None, indices=None, bands=None, dataset_extra_data=["N","P","K"],image_identifier="Point_ID"):
        """Creates the dataframe ready for the algorithm processing

        Args:
            subset (Dataframe): The filtered part of the Dataset.get_eopatches_dataframe() which are supposed to fit
            indices ([string], optional): The indices available in the dataset. Defaults to None.
            bands ([string], optional): The bands avaiable in the datase. Defaults to None.
            dataset_extra_data([string], optional): Information in the METADATA portion of the dataset that should be included in the parsed data.
            image_identifier(string, optional): The identifying column on the information in the METADATA portion of dataset.

        Returns:
            [type]: [description]
        """
        if not isinstance(subset, list) and subset is not None: raise TypeError("Subset must be a list of EOPatches")
        tmp_indices = INDICES[-1]
        if indices is None:
            indices = tmp_indices#+dataset_extra_data+[image_identifier]

        if bands is None:
            bands = BANDS

        if subset is not None:
            self.subset = subset
        
        self.indices = indices
        self.bands = bands
        self.dataset_extra_data = dataset_extra_data
        self.image_identifier = image_identifier
        
        #if self.save_dataframe is not None:
        #    return self.save_dataframe
        
        # all_features = indices+bands
        save_dataframe = pd.DataFrame(columns=indices+bands+dataset_extra_data+[image_identifier]) 
        # self.imagery = indices+len(dataset_extra_data)+1 #dataset_extra_data+image_identifier
        
        #for i,eopatch in enumerate(self.dataset.get_eopatches()[eopatches_indices]):
        excluded_patches = []
        for i,eopatch in enumerate(self.subset):
            try:
                mask = eopatch.get_masked_region()
                values = eopatch.get_values_of_masked_region(indices,bands)
                for extra_variable in dataset_extra_data:
                    values[extra_variable] = eopatch.get_dataset_entry_value(extra_variable,is_pixelized=True)
                
                values[image_identifier] = eopatch.get_dataset_entry_value(image_identifier,is_pixelized=True)
                values = pd.DataFrame(values)
                save_dataframe = save_dataframe.append(values, ignore_index=True)
            except:
                #no groudtruth for you :c
                excluded_patches.append(eopatch)

        self.save_dataframe = save_dataframe
        return self.save_dataframe
        
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

        if(self.save_dataframe is None):
            raise ExecError("Parser.create_dataframe must be called before")
        if(image_ids is None):
            vals = self.save_dataframe.round(2)
        else:
            vals = self.save_dataframe[self.save_dataframe[self.image_identifier].isin(image_ids)].round(2)
        if(features is None):
            x_values = vals[self.indices+self.bands].values.reshape(-1,len(self.indices+self.bands))
        else:
            x_values = vals[features].values.reshape(-1,len(features))
        y_values = vals[variable_to_fit].values.reshape(-1,1).ravel()        
        return (x_values,y_values)
    
    