import os
import pandas as pd
import numpy as np
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk,SaveTask, LoadTask
from eolearn.core.eoexecution import EOExecutor

from dataset.TSPatch import TSPatch
from ..utils import AddIndicesTask
class Dataset:
    def __init__(self, eopatches_folder):
        self.eopatches_folder = eopatches_folder
        self._eopatches = []
        for path in os.listdir(eopatches_folder):
            self._eopatches.append(TSPatch.load(os.path.join(eopatches_folder,path),lazy_loading=True))
    
    def get_eopatches(self):
        return self._eopatches

    eopatches = property(get_eopatches,None,None,"""List of EOPATCHES present in the given folder""")

    def get_eopatch(self,index):
        return self._eopatches[index];

    def add_indices_to_eopatches(self):
        load = LoadTask(self.eopatches_folder)
        add_indices = AddIndicesTask()
        execution_args = []
        save = SaveTask(self.eopatches_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        workflow = LinearWorkflow(load, add_indices,save)

        for i in range(0,len(os.listdir(self.eopatches_folder))):
            execution_args.append(
            {
                load: {'eopatch_folder': f'eopatch_{i}'},
                save: {'eopatch_folder': f'eopatch_{i}'}
            })
        executor = EOExecutor(workflow, execution_args, save_logs=True)
        executor.run(workers=5, multiprocess=False)
    
    def get_eopatches_dataframe(self):
    
        """Returns the DATASET information
                
        Returns:
            DataFrame: 
        """
        if self.dfeopatches != None:
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

    