from eolearn.core import EOTask,FeatureType
from eolearn.features import LinearInterpolation
import numpy as np


class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):        
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool), 
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))
    
class CountValid(EOTask):   
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """
    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name
        
    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name, np.count_nonzero(eopatch.mask[self.what],axis=0))
        
        return eopatch


class NormalizedDifferenceIndex(EOTask):   
    """
    The tasks calculates user defined Normalised Difference Index (NDI) between two bands A and B as:
    NDI = (A-B)/(A+B).
    """
    def __init__(self, feature_name, band_a, band_b):
        self.feature_name = feature_name
        self.band_a_fetaure_name = band_a.split('/')[0]
        self.band_b_fetaure_name = band_b.split('/')[0]
        self.band_a_fetaure_idx = int(band_a.split('/')[-1])
        self.band_b_fetaure_idx = int(band_b.split('/')[-1])
        
    def execute(self, eopatch):
        band_a = eopatch.data[self.band_a_fetaure_name][..., self.band_a_fetaure_idx]
        band_b = eopatch.data[self.band_b_fetaure_name][..., self.band_b_fetaure_idx]
        
        ndi = (band_a - band_b) / (band_a  + band_b)
        
        eopatch.add_feature(FeatureType.DATA, self.feature_name, ndi[..., np.newaxis])
        
        return eopatch

    
class EuclideanNorm(EOTask):   
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """
    def __init__(self, feature_name, in_feature_name):
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name
    
    def execute(self, eopatch):
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr**2, axis=-1))
        
        eopatch.add_feature(FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch

class InterpolationTask(EOTask):
    
    def execute(self,eopatch):
        
        start = eopatch.timestamp[0].strftime("%Y-%m-%d")
        end = eopatch.timestamp[-1].strftime("%Y-%m-%d")
        resampled_range = (start, end, 4)
        linear_interp = LinearInterpolation(
            'FEATURES', # name of field to interpolate
            mask_feature=(FeatureType.MASK_TIMELESS, 'IS_VALID'), # mask to be used in interpolation
            copy_features=[(FeatureType.DATA, 'NORM'),(FeatureType.MASK, 'IS_DATA'),(FeatureType.MASK, 'IS_VALID'),(FeatureType.MASK_TIMELESS, 'IS_VALID'),(FeatureType.VECTOR_TIMELESS,"LOCATION")], # features to keep            resample_range=resampled_range,
        )
        return linear_interp.execute(eopatch)

class AddIndicesTask(EOTask):
    def execute(self, eopatch):
        indices = ["RVI","NDVI74","IRECI","GM","GNDVI","RECHI","REPLI","S2REP","SRRE","NDVI_NB"]

        bands = eopatch.data["BANDS"]
        def band(band_id):
            t,w,h,_ = bands.shape
            return bands[...,band_id+1].reshape(t,w,h,1) 
        #band = lambda band_id: bands[...,band_id+1] 
        eopatch.data["NDVI74"] = (band(7) - band(4)) / (band(7) + band(4))
        eopatch.data["IRECI"] = (band(7) - band(4)) / (band(5) + band(6))
        eopatch.data["REM"] = (band(7) / band(5)) - 1
        eopatch.data["GM"] = (band(7) / band(3)) - 1
        rre = (band(7) + band(4)) / 2
        eopatch.data["REPLI"] = 700 + 40 * (rre - band(5)) / (band(6) - band(5))
        eopatch.data["RECHI"] = (band(6) - band(5))/band(5)
        eopatch.data["S2REP"] = 705 + 35 * ((rre - band(5)) / (band(6) - band(5)))
        eopatch.data["GNDVI"] = (band(8) - band(3)) / (band(8) + band(3))
        eopatch.data["SRRE"] = (band(5)) / (band(10))
        eopatch.data["NDVI_NB"] = (band(9)- band(4))/(band(9) + band(4))
        eopatch.data["RVI"] = (band(8) / band(4))
        for index in indices:
            np.nan_to_num(eopatch.data[index],False,nan=0,posinf=0,neginf=0)
        return eopatch        