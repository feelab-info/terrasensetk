from eolearn.core import EOPatch
import numpy as np
class TSPatch(EOPatch):

    def __init__(self):
        super().__init__()
        self.patch = super()

    def get_masked_region(self):
        mask = self.patch.mask_timeless["IS_VALID"].squeeze()
        mask_filtered = np.where(mask==5,0,mask)
        return mask_filtered

    def get_values_of_masked_region(self ,indices = None ,band_names = None,as_array=True):
        """Returns the pixels in the masked region for the selected indices and band_names for each of the 

        Args:
            indices ([type]): [description]
            band_names ([type]): [description]
            as_array (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        values = {}
        eopatch = self.patch
        masked_region = self.get_masked_region()
        for i in range(0,eopatch.data["BANDS"].shape[-1]):
            values[band_names[i]] = eopatch.data["BANDS"][-1,...,i]*masked_region
            if as_array: 
                values[band_names[i]] = values[band_names[i]][masked_region!=0]
                
        for i in indices:
            values[i] = eopatch.data[i][-1,...,-1]*masked_region
            if as_array:
                values[i] = values[i][masked_region!=0]
        return values

    def represent_image(self,estimation):
        eopatch = self.patch
        mask = self.get_masked_region()
        mask = mask.astype(float)
        image = eopatch.data["BANDS"][-1][...,[3,2,1]]
        _max = 255#dfeopatches["N"].max()
        _min = 0#dfeopatches["N"].min()
        convert_est = estimation/_max
        for i,tup in enumerate(zip(mask.nonzero()[0],mask.nonzero()[1])):
            x = tup[0]
            y = tup[1]
            try:
                image[x,y] = [convert_est[i], 0,0]
            except BaseException as err:
                print(f"The estimation most likely doesn't correspond to the eopatch: {err}")
        return image


    def get_dataset_entry_value(self,nutrient, is_pixelized = False):
        """[summary]

        Args:
            index (int): [description]
            nutrient (string): Entry of the table (N,P,K)
            is_pixelized (bool, optional): If true returns the value in the shape of the mask, if not, returns a scalar(int). Defaults to False.

        Returns:
            [type]: [description]
        """
        eopatch = self.patch
        mask = self.get_masked_region(eopatch)
        if is_pixelized:
            return eopatch.vector_timeless["LOCATION"][nutrient].values[0]*mask[mask!=0]
        return eopatch.vector_timeless["LOCATION"][nutrient].values[0]

    def get_eopatch_mask(self,index, include_indices = True):
        
        mask_filtered = self.get_masked_region();
        if include_indices : return (mask_filtered,mask_filtered.nonzero())
        return mask_filtered