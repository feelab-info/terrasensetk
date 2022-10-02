import os
import pandas as pd
import numpy as np
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk,SaveTask, LoadTask
from eolearn.core.eoexecution import EOExecutor

from .TSPatch import TSPatch
from ..utils import AddIndicesTask
class Dataset:
    def __init__(self, eopatches_folder):
        self.eopatches_folder = eopatches_folder
        self._eopatches = []
        
        for path in os.listdir(eopatches_folder):
            self._eopatches.append(TSPatch.load(os.path.join(eopatches_folder,path),lazy_loading=True))
        #self.index_dic = self._eopatches[0].data
        self.index_dic = {}

    def get_eopatches(self):
        return self._eopatches

    eopatches = property(get_eopatches,None,None,"""List of EOPATCHES present in the given folder""")

    def get_eopatch(self,index):
        return self._eopatches[index]

    def add_index(self,index_name, index_formula):
        """Adds a specific index to the dataset in question
        
        NOTE:To make the index persist `save_indices_to_patches` must be called

        Args:
            index_name (str): The name of the index
            index_formula (str): The formula for the specified index

        Example:

            >>>dataset.add_index("NDVI","(B07-B04)/(B07+B04)")

            >>>dataset.add_index("IRECI","(B07-B04)/(B05/B06)")

            >>>dataset.save_indices_to_patches()

        """
        self.index_dic[index_name] = index_formula


        
    def save_indices_to_eopatches(self):
        """Saves the previously indicated indices into the dataset.

        Example:

            >>>dataset.add_index("NDVI","(B07-B04)/(B07+B04)")
            >>>dataset.add_index("IRECI","(B07-B04)/(B05/B06)")
            >>>dataset.save_indices_to_patches()
        """
        load = LoadTask(self.eopatches_folder)
        
        available_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

        add_indices = AddIndicesTask(self.index_dic,available_bands)
        
        execution_args = []
        save = SaveTask(self.eopatches_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        workflow = LinearWorkflow(load, add_indices,save)
        eopatch_folders = os.listdir(self.eopatches_folder)
        print(len(eopatch_folders))
        for i in eopatch_folders:
            execution_args.append(
            {
                load: {'eopatch_folder': f'{i}'},
                add_indices: {},
                save: {'eopatch_folder': f'{i}'}
            })
        executor = EOExecutor(workflow, execution_args, save_logs=True)
        executor.run(workers=5, multiprocess=False)
        executor.make_report()

    
    def get_eopatches_dataframe(self):
    
        """Returns the DATASET information
                
        Returns:
            DataFrame: 
        """
        if self.dfeopatches is not None:
            return self.dfeopatches
        tmp_arr = []
        cols = self._eopatches[-1].vector_timeless["LOCATION"].columns.values.tolist()
        cols.append("EOPATCH_ID")
        for i, eopatch in enumerate(self._eopatches):
            _tmp = eopatch.vector_timeless["LOCATION"].copy(deep=True).values.tolist()[-1]
            _tmp.append(i)
            tmp_arr.append(_tmp)
        self.dfeopatches = pd.DataFrame(tmp_arr,columns = cols)
        return self.dfeopatches

    