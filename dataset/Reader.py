import sentinelhub as sh
import geopandas as gdp
import matplotlib.pyplot as plt
import ArgChecker.ArgChecker as type_check
    """
        change location of lucas from meta_info to vector
        
        Get the images
        Define which bboxes have groundtruth

        Define the pipeline for writing the eopatches
            allow for setting the bands needed
            allow for adding extra tasks?
            allow for adding addicional indexes
        
        add type check to arguments
        need to define the lucas_copernicus csv
    """
class Reader():

    def __init__(self,shapefile,bands, country, config=null,eopatch_size = 500):
        self._shapefile = shapefile
        self._bands = bands
        self._country = country #should change to region
        self._eopatch_size = eopatch_size;
        self._world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        dataset = world[world.name==self.country]
        self.dataset = dataset.to_crs(sh.CRS.WGS84.pyproj_crs())
       
    def get_groundtruth():
        """[summary]

        Returns:
            GeoDataFrame: Contains the groundtruth data within the given region
        """
        if self._groundtruth_points != null:
            return self._groundtruth_points
        self._groundtruth_points = gpd.sjoin(self.ground_truth,self.dataset,op="within",how="left").drop("index_right", axis="columns")
        return self._groundtruth_points

    def get_bbox_with_data():
        """[summary]

        Returns:
            GeoDataFrame: Contains the bboxes which have associated groundtruth
        """
        if self._bbox_with_groundtruth != null:
            return self._bbox_with_groundtruth

        self._bbox_with_groundtruth = gpd.sjoin(self.get_bbox(),self.get_groundtruth(),op='overlaps',how='inner')
        self._bbox_with_groundtruth.append(gpd.sjoin(self.get_bbox(),self.get_groundtruth(),op='contains',how='inner'))
        return self._bbox_with_groundtruth
    
    @type_check()
    def get_bbox(self,expected_bbox_size=2000:int,reset=False:bool):

        """
        Creates a grid of bbox over the dataset

        Args:
            dataset (GeoDataFrame): [description]
            expected_bbox_size (int, optional): The desired size of the bbox in meters. Defaults to 2000.
            reset (bool, optional): Wether it should recalculate the bbox_list. Defaults to False.

        Returns:
            GeoDataFrame: GeoDataFrame of the dataset divided in square bbox of size of ´expected_bbox_size´.
        """

        if(self._dataset_bbox != null || reset == False):
             return self._dataset_bbox

        dataset_shape = self.dataset.to_crs("EPSG:3395").geometry.values[0]

        height = dataset_shape.bounds[2] - dataset_shape.bounds[0]
        width = dataset_shape.bounds[3] - dataset_shape.bounds[1]
        
        bbox_num_y =  int(height/expected_bbox_size)
        bbox_num_x =  int(width/expected_bbox_size)

        dataset_bbox_splitter = sh.BBoxSplitter(self.dataset.geometry.to_list(),sh.CRS.WGS84,(bbox_num_y,bbox_num_x))

        geometry = [Polygon(bbox.get_polygon()) for bbox in dataset_bbox_splitter.get_bbox_list()]
        self._dataset_bbox = gpd.GeoDataFrame(crs=sh.CRS.WGS84, geometry=geometry)
        return self._dataset_bbox
    
    #should probably be in a utils module
    @staticmethod
    def get_time_interval(middle_date, number_of_days):
        """
        Gets the time interval surrounding the middle date separated by slashes
        Args:
            middle_date: A string containing the date which will be included in the timerange
            number_of_days: The number of days counting from the `middle_date` that will correspond to the min and max date
            
        Returns:
            A list with the ´number_of_days´ before and after of the ´middle_date´
            
        Example:
            >>> get_time_interval("15/09/1998", 3)
            >>> ['1998-09-12', '1998-09-18']
            
        """
        point_date = datetime.datetime.strptime(middle_date, '%d/%m/%y')
        days_before = point_date - datetime.timedelta(days=number_of_days)
        days_after = point_date + datetime.timedelta(days=number_of_days)
        return [days_before.strftime('%Y-%m-%d'), days_after.strftime('%Y-%m-%d')]

    def plot_dataset(self,save_img=null):
        """Plots the existing information in matplotlib

        Args:
            save_img (string, optional): Path to where the plot should be saved. Defaults to null.
        """
        plt, ax = plt.subplots(figsize=(400,400))
        self._dataset_bbox.plot(ax=ax,facecolor='w',edgecolor='r', alpha=0.4)
        self.dataset(ax=ax, facecolor='w', edgecolor='b', alpha=0.5)
        if(self.ground_truth_bbox != null):
             self.ground_truth_bbox.plot(ax=ax, facecolor='g', edgecolor='black',alpha=0.7)
        if(save_img != null):
            plt.savefig(os.path.join(save_img,self._country+".png")
        else:
            plt.show()

        

