from eolearn.core import EOPatch
import numpy as np
import datetime as dt
class TSPatch(EOPatch):
    """Extends the functionality of the original eo-patch implementation with methods to include extra functionality

    Args:
        EOPatch (EOPatch): eo-learn abstraction to represent a single region 
    """
    def __init__(self, eopatch = None):
        if eopatch is not None:
            self.patch=eopatch
            return
        super().__init__()
        self.patch = super()
    
    def __getdata(self):
        return self.patch.data
        
    def __getmask(self):
        return self.patch.mask
        
    def __getscalar(self):
        return self.patch.scalar
        
    def __getlabel(self):
        return self.patch.label
        
    def __getvector(self):
        return self.patch.vector
        
    def __getdata_timeless(self):
        return self.patch.data_timeless
        
    def __getmask_timeless(self):
        return self.patch.mask_timeless
        
    def __getscalar_timeless(self):
        return self.patch.scalar_timeless
        
    def __getlabel_timeless(self):
        return self.patch.label_timeless
        
    def __getvector_timeless(self):
        return self.patch.vector_timeless
        
    def __getmeta_info(self):
        return self.patch.meta_info
        
    def __getbbox(self):
        return self.patch.bbox
    
    def __gettimestamp(self):
        return self.patch.timestamp
        
    def __setdata(self,value):
        self.patch.data = value
        
    def __setmask(self,value):
        self.patch.mask = value
        
    def __setscalar(self,value):
        self.patch.scalar = value
        
    def __setlabel(self,value):
        self.patch.label = value
        
    def __setvector(self,value):
        self.patch.vector = value
        
    def __setdata_timeless(self,value):
        self.patch.data_timeless = value
        
    def __setmask_timeless(self,value):
        self.patch.mask_timeless = value
        
    def __setscalar_timeless(self,value):
        self.patch.scalar_timeless = value
        
    def __setlabel_timeless(self,value):
        self.patch.label_timeless = value
        
    def __setvector_timeless(self,value):
        self.patch.vector_timeless = value
        
    def __setmeta_info(self,value):
        self.patch.meta_info = value
        
    def __setbbox(self,value):
        self.patch.bbox = value
        
    def __settimestamp(self,value):
        self.patch.timestamp = value

    data = property(__getdata,__setdata,None,"""The data feature of the eopatch.""")
    mask = property(__getmask,__setmask,None,"""The mask feature of the eopatch.""")
    scalar = property(__getscalar,__setscalar,None,"""The scalar feature of the eopatch.""")
    label = property(__getlabel,__setlabel,None,"""The label feature of the eopatch.""")
    vector = property(__getvector,__setvector,None,"""The vector feature of the eopatch.""")
    data_timeless = property(__getdata_timeless,__setdata_timeless,None,"""The data_timeless feature of the eopatch.""")
    mask_timeless = property(__getmask_timeless,__setmask_timeless,None,"""The mask_timeless feature of the eopatch.""")
    scalar_timeless = property(__getscalar_timeless,__setscalar_timeless,None,"""The scalar_timeless feature of the eopatch.""")
    label_timeless = property(__getlabel_timeless,__setlabel_timeless,None,"""The label_timeless feature of the eopatch.""")
    vector_timeless = property(__getvector_timeless,__setvector_timeless,None,"""The vector_timeless feature of the eopatch.""")
    meta_info = property(__getmeta_info,__setmeta_info,None,"""The meta_info feature of the eopatch.""")
    bbox = property(__getbbox,__setbbox,None,"""The bbox feature of the eopatch.""")
    timestamp = property(__gettimestamp,__settimestamp,None,"""The timestamp feature of the eopatch.""")


    def get_masked_region(self):
        mask = self.patch.mask_timeless["IS_VALID"].squeeze()
        mask_filtered = np.where(mask==5,0,mask)
        return mask_filtered

    def get_values_of_masked_region(self ,indices = None ,band_names = None,as_array=True):
        """Returns the pixels in the masked region for the selected `indices` and `band_names` for each of the patch

        Args:
            indices (str): The list of indices in which we want to get the values of
            band_names (str): The list of indices in which we want to get the values of
            as_array (bool, optional): If true returns in 1D array form(only the values with data), else returns in 2D array. Defaults to True.

        Returns:
            ndarray: If `as_array` is true returns in 1D array form(only the values with data), else returns in 2D array.
        """
        values = {}
        eopatch = self.patch
        masked_region = self.get_masked_region()
        nearest_image_index = self._get_index_nearest_to_collection_date()
        for i in range(0,eopatch.data["BANDS"].shape[-1]):
            values[band_names[i]] = eopatch.data["BANDS"][nearest_image_index,...,i]*masked_region
            if as_array: 
                values[band_names[i]] = values[band_names[i]][masked_region!=0]
        
        for i in indices:
            values[i] = eopatch.data[i][nearest_image_index,...,-1]*masked_region
            if as_array:
                values[i] = values[i][masked_region!=0]
        return values

    def represent_image(self,estimation):
        """Draws an image with the values estimated

        Args:
            estimation (array): Array with the size of the masked region(1D)

        Returns:
            array: 2D image
        """
        nearest_image_index = self._get_index_nearest_to_collection_date()

        eopatch = self.patch
        mask = self.get_masked_region()
        mask = mask.astype(float)
        image = eopatch.data["BANDS"][nearest_image_index][...,[3,2,1]]
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
        """Returns the value of column in the dataset

        Args:
            nutrient (string): Entry of the table (N,P,K, or other)
            is_pixelized (bool, optional): If true returns the value in the shape of the mask, if not, returns a scalar(int). Defaults to False.

        Returns:
            value|array: If `is_pixelized` is true returns the value in the shape of the mask, if not, returns a scalar(int).
        
        Example:

        >>> patch.get_dataset_entry_value("N",True)
            [1.9,1.9,1.9,...,1.9,]

        >>> patch.get_dataset_entry_value("N",False)
            1.9
        """
        eopatch = self.patch
        mask = self.get_masked_region()
        if is_pixelized:
            #return eopatch.vector_timeless["LOCATION"][nutrient].values[0]*mask[mask!=0]
            return [eopatch.vector_timeless["LOCATION"][nutrient].values[0] for i in range(0,len(mask[mask!=0]))]
        return eopatch.vector_timeless["LOCATION"][nutrient].values[0]

    def get_eopatch_mask(self, include_indices = True):
        """Can't remember what this does

        Args:
            include_indices (bool, optional): Defaults to True.

        Returns:
            array: ??
        """

        mask_filtered = self.get_masked_region()
        if include_indices : return (mask_filtered,mask_filtered.nonzero())
        return mask_filtered
    
    @classmethod
    def load(cls, path, lazy_loading=True):
        eopatch = super().load(path,lazy_loading=lazy_loading)
        return TSPatch(eopatch)
    
    def _get_index_nearest_to_collection_date(self):
        smallest_index = 0
        smallest_difference = dt.timedelta(days=2000)
        try:
            collected_day = dt.datetime.strptime(self.get_dataset_entry_value("SURVEY_DATE"),'%d/%m/%y')
        except:
            return -1
        for i,image_date in enumerate(self.timestamp):
            current_difference = abs(collected_day - image_date)
            if(current_difference < smallest_difference):
                smallest_difference = current_difference
                smallest_index = i
        return smallest_index
