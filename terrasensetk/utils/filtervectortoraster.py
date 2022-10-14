from eolearn.geometry import VectorToRaster
from eolearn.core import EOTask


class FilterVectorToRaster(EOTask):
    
    def __init__(self,raster_feature,values,raster_resolution):
        self.featureType = raster_feature
        self.values = values
        self.raster_resolution = raster_resolution
        
    def execute(self,eopatch,dataset):
        vector = dataset[dataset.geometry.intersects(eopatch.bbox.geometry)]
        for i,vec in vector.iterrows():
            feature_name = self.featureType[-1]+"_"+str(vec.POINT_ID)
            feature = (self.featureType[0], feature_name)
            self.vector2rastertask = VectorToRaster(vector,raster_feature=feature,values=self.values,raster_resolution=self.raster_resolution)
            eopatch = self.vector2rastertask.execute(eopatch)
        return eopatch