class Reader():

    def __init__(self,shapefile,bands,country, config=null,eopatch_size = 500):
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
    