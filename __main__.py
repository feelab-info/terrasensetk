import utils,dataset
import geopandas as gpd

country = dataset.Reader.Reader(country="Portugal")
country.get_groundtruth()

country.get_bbox_with_data()
country.plot_dataset()
