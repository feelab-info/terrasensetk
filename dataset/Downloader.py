from sentinelhub.areas import CustomGridSplitter
from ..utils.eotasks import CountValid, EuclideanNorm, SentinelHubValidData
from ..utils.utils import get_lucas_copernicus_path, get_time_interval,to_square
from ..utils.filtervectortoraster import FilterVectorToRaster
import itertools

import sentinelhub as sh
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
import sys
import os
import datetime
import numpy as np
from eolearn.io.processing_api import SentinelHubInputTask
from sentinelhub import BBox, DataCollection,BBoxSplitter,CustomGridSplitter,BBoxCollection
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor, ExtractBandsTask, MergeFeatureTask, AddFeature
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask
from eolearn.geometry import VectorToRaster
from rasterio.enums import MergeAlg
import math
import shapely
import compress_pickle as cpickle

class Downloader:
    """
    Class to download a dataset based on Lucas Copernicus
    
    """

    def __init__(self,shapefile = None,bands = None, country = None, continent = None, config=None):~
        
        self._init_classvars()
        if(shapefile is not None):
            self.dataset = gpd.read_file(shapefile)
        else:
            self._world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            if(country is not None):
                self.dataset = self._world[self._world.name==country]
            elif(continent is not None):
                self.dataset = self._world[self._world.continent==continent].dissolve(by="continent")
            else:
                ValueError("Either the shapefile or country must be provided")

        self._bands = bands
        #self._eopatch_size = eopatch_size
        self.dataset = self.dataset.to_crs(sh.CRS.WGS84.pyproj_crs())

    
    @classmethod
    def from_pickle(self,file):
        """Loads the pickled Downloader object

        Returns:
            Downloader: An instance that was saved in the pickle file
        """
        with open(file,'rb') as f:
            instance = cpickle.load(f,compression="bz2")
            return instance


        
    def to_pickle(self,file):
        """ Pickles the current Downloader object that can be loaded by the ´from_pickle´ function
        """
        with open(file,'wb') as f:
            cpickle.dump(self,f,compression="bz2")


    def get_groundtruth(self,crop = None):
        """Returns the groundtruth example in the LUCAS Copernicus dataset
        Args:
            crop: which crop to filter by possible values: 
                ['Grassland without tree/shrub cover',
                'Shrubland without tree cover', 'Peatbogs', 'Inland marshes',
                'Spontaneously vegetated surfaces', 'Rape and turnip rape',
                'Barley', 'Spruce dominated mixed woodland',
                'Grassland with sparse tree/shrub cover', 'Broadleaved woodland',
                'Temporary grasslands', 'Spruce dominated coniferous woodland',
                'Oats', 'Clovers', 'Sunflower',
                'Pine dominated coniferous woodland', 'Other mixed woodland',
                'Common wheat', 'Other coniferous woodland',
                'Pine dominated mixed woodland',
                'Shrubland with sparse tree cover', 'Sugar beet', 'Lucerne', 'Rye',
                'Potatoes', 'Maize', 'Triticale',
                'Other fibre and oleaginous crops', 'Other root crops',
                'Apple fruit', 'Mixed cereals for fodder',
                'Other artificial areas', 'Durum wheat', 'Dry pulses', 'Vineyards',
                'Other bare soil', 'Olive groves', 'Nurseries', 'Oranges',
                'Tomatoes', 'Other leguminous and mixtures for fodder', 'Soya',
                'Other fruit trees and berries', 'Inland fresh running water',
                'Nuts trees', 'Pear fruit', 'Other non-permanent industrial crops',
                'Permanent industrial crops', 'Non built-up linear features',
                'Sand', 'Cotton', 'Cherry fruit', 'Other fresh vegetables',
                'Non built-up area features', 'Lichens and moss',
                'Rocks and stones', 'Other cereals', 'Strawberries',
                'Floriculture and ornamental plants', 'Other citrus fruit', 'Rice',
                'Buildings with 1 to 3 floors', 'Tobacco']
        Returns:
            GeoDataFrame: Contains the groundtruth data within the given region
        """
        if self._groundtruth_points is not None:
            return self._groundtruth_points
        groundtruth_gdf = pd.read_pickle(get_lucas_copernicus_path(),compression='bz2')
        
        if crop is not None:
            groundtruth_gdf = groundtruth_gdf[groundtruth_gdf.LC1_LABEL == crop]
        
        self._groundtruth_points = gpd.sjoin(groundtruth_gdf,self.dataset,op="within",how="inner").drop("index_right", axis="columns")
        return self._groundtruth_points


    def get_bbox_with_data(self):
        """

        Returns:
            GeoDataFrame: Contains the bboxes which have associated groundtruth
        """
        if self._bbox_with_groundtruth is not None:
            return self._bbox_with_groundtruth

        self._bbox_with_groundtruth = gpd.sjoin(self.get_bbox(),self.get_groundtruth(),op='overlaps',how='inner')
        self._bbox_with_groundtruth = self._bbox_with_groundtruth.append(gpd.sjoin(self.get_bbox(),self.get_groundtruth(),op='contains',how='inner'))
        return self._bbox_with_groundtruth
    
   
    def get_bbox(self,expected_bbox_size=2000,reset=False):

        """
        Creates a grid of bbox over the dataset

        Args:
            dataset (GeoDataFrame): [description]
            expected_bbox_size (int, optional): The desired size of the bbox in meters. Defaults to 2000.
            reset (bool, optional): Wether it should recalculate the bbox_list. Defaults to False.

        Returns:
            GeoDataFrame: GeoDataFrame of the dataset divided in square bbox of size of ´expected_bbox_size´.
        """
        

        dataset_shape = self.dataset.to_crs("EPSG:3395").geometry.values[-1]

        width = math.ceil(dataset_shape.bounds[2] - dataset_shape.bounds[0])
        height = math.ceil(dataset_shape.bounds[3] - dataset_shape.bounds[1])
        bbox_num_y =  math.ceil(height/expected_bbox_size)
        bbox_num_x =  math.ceil(width/expected_bbox_size)
        # print(f"bbox_y: {bbox_num_y} bbox_xa:{bbox_num_x}")
        # print(f"width: {width} height: {height}")
        
        #create bboxes around the groundtruth
        self.dataset_bbox_splitter = CustomGridSplitter([shapely.geometry.MultiPolygon([i.buffer(0.002) for i in self.get_groundtruth().geometry.values])],
            sh.CRS.WGS84.pyproj_crs(),
            BBoxCollection(self.get_groundtruth().geometry.apply(to_square).apply(lambda x: BBox(x,sh.CRS.WGS84)).to_list()))
        geometry = [Polygon(bbox.get_polygon()) for bbox in self.dataset_bbox_splitter.get_bbox_list()]
        self._dataset_bbox = gpd.GeoDataFrame(crs=sh.CRS.WGS84.pyproj_crs(), geometry=geometry)
        return self._dataset_bbox
    

    def plot_dataset(self,save_img=None):
        """Plots the existing information in matplotlib

        Args:
            save_img (string, optional): Path to where the plot should be saved. Defaults to None.
        """
        fig, ax = plt.subplots(figsize=(30,30))
        self._dataset_bbox.plot(ax=ax,facecolor='w',edgecolor='r', alpha=0.4)
        self.dataset.plot(ax=ax, facecolor='w', edgecolor='b', alpha=0.5)
        self.get_groundtruth().plot(ax=ax, facecolor='b', alpha=0.5)
        if(self._bbox_with_groundtruth is not None):
            self._bbox_with_groundtruth.plot(ax=ax, facecolor='g', edgecolor='black',alpha=0.7)
        if save_img is not None:
            plt.savefig(os.path.join(save_img,"country.png"))
        else:
            plt.show()


    def _init_classvars(self):
        self.dataset = None
        self._groundtruth_points = None
        self._dataset_bbox = None
        self._bbox_with_groundtruth = None
        self._ground_truth = None        
    
    def download_images(self,path,subset=None):
        """Downloads the specified images into the users filesystem.

        Args:
            path (str): Path to where the dataset should be saved
            subset (DataFrame, optional): Slice of the dataframe returned by `get_bbox_with_data()`.
        """
        if subset is None:
            subset = self.get_bbox_with_data()
        if not os.path.isdir(path):
            os.makedirs(path)

        add_data = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, 'BANDS'),
            resolution=10,
            #bands=['B02','B03','B04','B08'],
            maxcc=0.8,
            time_difference=datetime.timedelta(minutes=120),
            data_collection=DataCollection.SENTINEL2_L1C,
            additional_data=[(FeatureType.MASK, 'dataMask', 'IS_DATA'),
                     (FeatureType.MASK, 'CLM'),
                     (FeatureType.DATA, 'CLP')],
            max_threads=5
        )
        
        save = SaveTask(path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        add_vector = AddFeature((FeatureType.VECTOR_TIMELESS,"LOCATION"))

        #add_lucas = AddFeature((FeatureType.META_INFO,"LUCAS_DATA"))
        #to get the surrounding data, one can apply a buffered vector to raster and set the non overlapped value some value to distinguish
        add_raster_buffer = VectorToRaster((FeatureType.VECTOR_TIMELESS,"LOCATION"),(FeatureType.MASK_TIMELESS,"IS_VALID"), values = 5,buffer=0.0005, raster_shape=(FeatureType.MASK, 'IS_DATA'),no_data_value=0,raster_dtype=np.uint8)
        add_raster = VectorToRaster((FeatureType.VECTOR_TIMELESS,"LOCATION"),(FeatureType.MASK_TIMELESS,"IS_VALID"), values = 1,raster_shape=(FeatureType.MASK, 'IS_DATA'),write_to_existing = True,no_data_value=0,raster_dtype=np.uint8)
        

        norm = EuclideanNorm('NORM','BANDS')

        add_sh_valmask = AddValidDataMaskTask(SentinelHubValidData(),'IS_VALID')
        add_valid_count = CountValid('IS_VALID', 'VALID_COUNT')

        concatenate = MergeFeatureTask({FeatureType.DATA: ['BANDS']},(FeatureType.DATA, 'FEATURES'))
        workflow = LinearWorkflow(add_data,add_vector,add_raster_buffer,add_raster,norm,add_sh_valmask,add_valid_count,concatenate,save)

        execution_args = []
        for id, wrap_bbox in enumerate(subset.iterrows()):
            i, bbox = wrap_bbox

            time_interval = (get_time_interval(bbox.SURVEY_DATE,5))
            gdf = gpd.GeoDataFrame(bbox,crs=sh.CRS.WGS84.pyproj_crs())
            gdf = gdf.transpose()
            gdf = gdf.rename(columns={0:'geometry'}).set_geometry('geometry')
            gdf.set_geometry('geometry')

            lucas_points_intersection = self.get_groundtruth()[self.get_groundtruth().geometry.values.intersects(gdf.geometry.values[0])]
            execution_args.append({
                add_vector:{'data': lucas_points_intersection},
                add_data:{'bbox': BBox(bbox.geometry,crs=self.dataset.crs), 'time_interval': time_interval},
                save: {'eopatch_folder': f'eopatch_{id}'}
            })
        executor = EOExecutor(workflow, execution_args, save_logs=True)
        executor.run(workers=5, multiprocess=False)

        executor.make_report()
