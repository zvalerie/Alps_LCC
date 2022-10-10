import os
from osgeo import gdal
import geopandas as gpd
import subprocess
from tqdm import tqdm

def cropTiles(label_path, grid_path, tmp_path):
    '''
    crop the label(.tif) into tiles
    resolution: 100m x 100m
    '''
    # read grid shp
    grid = gpd.read_file(grid_path)
    
    for _, grid in tqdm(grid.iterrows()):
        tile_path= "/data/xiaolong/mask/" + grid.ID + "_label.tif"
        if os.path.isfile(tile_path):
            continue
        else:
            s = gpd.GeoSeries(grid.geometry)
            s.to_file(tmp_path)
            cmd = "gdalwarp -cutline " + tmp_path + " -tr 0.5 0.5 -crop_to_cutline " + label_path + " " + tile_path
            # print(type(cmd))
            _ = subprocess.call(cmd, shell = True, stdout= subprocess.DEVNULL, stderr=subprocess.STDOUT)
            # subprocess.run(cmd)
              
if __name__ == '__main__':
    label_path = "/data/xiaolong/label/Label.tif"
    grid_path = "/data/xiaolong/grid/aoiGrid100m.shp"
    tmp_path = "/data/xiaolong/tmp/tmp.shp"
    cropTiles(label_path, grid_path, tmp_path)