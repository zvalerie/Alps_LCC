import os
from osgeo import gdal
import geopandas as gpd
import subprocess
from tqdm import tqdm

shp = gpd.read_file('/data/xiaolong/mask_shp/Label.shp')
print(shp[0:50]['CLASS'])