import sentinelhub as sh
import geopandas as gdp
class Reader():

    def __init__(self,shapefile,bands, country, config=null,eopatch_size = 500):
        self._shapefile = shapefile
        self._bands = bands
        self._country = country
        self._eopatch_size = eopatch_size;
        self._world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        dataset = world[world.name==self.country]
        """
            See which library has the shapefiles, create a dictionary with the shapes and a key for each country
            Figure out a way how to create bboxes for all countries(maybe get the length and aim for 500m eopatches)
            change location of lucas from meta_info to vector
            Get the images
        """
    
    def get_bbox(self, dataset,expected_bbox_size=2000,reset=False):
        """Creates a grid of bbox over the dataset

        Args:
            dataset (GeoDataFrame): [description]
            expected_bbox_size (int, optional): The desired size of the bbox in meters. Defaults to 2000.
            reset (bool, optional): Wether it should recalculate the bbox_list. Defaults to False.

        Returns:
            GeoDataFrame: GeoDataFrame of the dataset divided in square bbox of size of ´expected_bbox_size´.
        """
        if(self._dataset_bbox != null || reset == False) return self._dataset_bbox

        dataset_shape = dataset.to_crs("EPSG:3395").geometry.values[0]

        height = dataset_shape.bounds[2] - dataset_shape.bounds[0]
        width = dataset_shape.bounds[3] - dataset_shape.bounds[1]
        
        bbox_num_y =  int(height/expected_bbox_size)
        bbox_num_x =  int(width/expected_bbox_size)

        dataset_bbox_splitter = sh.BBoxSplitter(dataset.geometry.to_list(),sh.CRS.WGS84,(bbox_num_y,bbox_num_x))

        geometry = [Polygon(bbox.get_polygon()) for bbox in dataset_bbox_splitter.get_bbox_list()]
        self._dataset_bbox = gpd.GeoDataFrame(crs=sh.CRS.WGS84, geometry=geometry)
        return self._dataset_bbox
        
    def plot_dataset(self):
        
        

