import dataset
import geopandas as gpd
print("1")
country = dataset.Reader.Reader(country="Portugal")
print("2")

country.get_groundtruth()
print("3")

country.get_bbox_with_data()
print("4")

country.download_images("C:/eopatchess")
print("5")

country.plot_dataset()
