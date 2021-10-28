import os
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk,SaveTask, LoadTask
from eolearn.core.eoexecution import EOExecutor
from ..utils import AddIndicesTask
class Reader:
    def __init__(self, eopatches_folder):
        self.eopatches_folder = eopatches_folder
        self._eopatches = []
        for path in os.listdir(eopatches_folder):
            self._eopatches.append(EOPatch.load(os.path.join(eopatches_folder,path),lazy_loading=True))
    
    def get_eopatches(self):
        return self._eopatches
    eopatches = property(get_eopatches,None,None,"""List of EOPATCHES present in the given folder""")

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
    