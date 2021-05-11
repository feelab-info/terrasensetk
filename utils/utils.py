from os import path

def get_lucas_copernicus_path():
    lucas_path = path.dirname(__file__)
    return path.join(lucas_path,"lucas_ds.pkl")