import sentinelhub as sh
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
import sys
import os
import datetime

from ..utils import get_lucas_copernicus_path, get_time_interval, FilterVectorToRaster
from eolearn.io.processing_api import SentinelHubInputTask
from sentinelhub import BBox, DataCollection
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor, ExtractBandsTask, MergeFeatureTask, AddFeature
#from .utils from ArgChecker
"""
        change location of lucas from meta_info to vector
        
        Get the images
        Define the pipeline for writing the eopatches
            allow for setting the bands needed
            allow for adding extra tasks?
            allow for adding addicional indexes
        
        add type check to arguments
        need to define the lucas_copernicus csv
    """
class Reader:


    def __init__(self,shapefile = None,bands = None, country = None, config=None, eopatch_size = 500):
        self._init_classvars()
        if(shapefile is not None):
            self.dataset = gpd.read_file(shapefile)
        elif(country is not None):
            self._world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            self.dataset = self._world[self._world.name==country]
        else:
            ValueError("Either the shapefile or country must be provided")

        self._bands = bands
        self._eopatch_size = eopatch_size
        self.dataset = self.dataset.to_crs(sh.CRS.WGS84.pyproj_crs())
    

    def get_groundtruth(self):
        """

        Returns:
            GeoDataFrame: Contains the groundtruth data within the given region
        """
        if self._groundtruth_points is not None:
            return self._groundtruth_points

        groundtruth_gdf = pd.read_pickle(get_lucas_copernicus_path(),compression='bz2')
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

        if(self._dataset_bbox is not None or reset != False):
            return self._dataset_bbox

        dataset_shape = self.dataset.to_crs("EPSG:3395").geometry.values[0]

        height = dataset_shape.bounds[2] - dataset_shape.bounds[0]
        width = dataset_shape.bounds[3] - dataset_shape.bounds[1]
        
        bbox_num_y =  int(height/expected_bbox_size)
        bbox_num_x =  int(width/expected_bbox_size)

        dataset_bbox_splitter = sh.BBoxSplitter(self.dataset.geometry.to_list(),sh.CRS.WGS84.pyproj_crs(),(bbox_num_y,bbox_num_x))

        geometry = [Polygon(bbox.get_polygon()) for bbox in dataset_bbox_splitter.get_bbox_list()]
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
    
    def download_images(self,path):
        if not os.path.isdir(path):
            os.makedirs(path)

        add_data = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, 'BANDS'),
            resolution=10,
            maxcc=0.8,
            time_difference=datetime.timedelta(minutes=120),
            data_collection=DataCollection.SENTINEL2_L1C,
            max_threads=5
        )
        
        save = SaveTask(path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
        add_lucas = AddFeature((FeatureType.META_INFO,"LUCAS_DATA"))
        add_lucas_raster = FilterVectorToRaster(raster_feature=(FeatureType.MASK_TIMELESS,"GROUND_TRUTH"),values=1,raster_resolution=10)

        workflow = LinearWorkflow(add_data,add_lucas,add_lucas_raster,save)

        execution_args = []
        for id, wrap_bbox in enumerate(self.get_bbox_with_data().head().iterrows()):
            i, bbox = wrap_bbox

            #lucas_points_intersection = portugal_gdf[portugal_gdf.intersects(bbox)]
            #time_interval = []
            #for point in bbox.SURVEY_DATE:
            time_interval = (get_time_interval(bbox.SURVEY_DATE,5))
            execution_args.append({
                add_data:{'bbox': BBox(bbox.geometry,crs=self.dataset.crs), 'time_interval': time_interval},
                add_lucas:{'data': bbox.drop("geometry")},
                add_lucas_raster:{'dataset':bbox.geometry},
                save: {'eopatch_folder': f'eopatch_{id}'}
            })
        executor = EOExecutor(workflow, execution_args, save_logs=True)
        executor.run(workers=5, multiprocess=False)

        executor.make_report()
