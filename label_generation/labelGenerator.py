import os
from pickle import NONE
from turtle import update
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

def vector2raster(input_shp_name, output_raster_name, pixel_size, no_data_value = 0, rdtype = gdal.GDT_Float32, **kwargs):
    '''
    transfer vector to raster
    '''
    # open data source 
    drv =ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = drv.Open(input_shp_name, 1)
    shp_layer = shp_ds.GetLayer()
    # read extent
    xMin, xMax, yMin, yMax = shp_layer.GetExtent()
    
    # get x and y resolution
    xRes = int((xMax - xMin) / pixel_size)
    yRes = int((yMax - yMin) / pixel_size)
    
    # creat raster
    driver = gdal.GetDriverByName('GTiff')
    raster_ds = driver.Create(output_raster_name, xRes, yRes, 1, rdtype)
    raster_ds.SetGeoTransform((xMin, pixel_size, 0, yMax, 0, -pixel_size))
    band = raster_ds.GetRasterBand(1)
    band.Fill(no_data_value)
    band.SetNoDataValue(no_data_value)

    # assign spatial reference to raster
    shp_sr = get_sr(shp_layer)
    raster_ds.SetProjection(shp_sr.ExportToWkt())
    
    # Rasterize
    gdal.RasterizeLayer(raster_ds, [1], shp_layer, options=["ALL_TOUCHED=TRUE", "ATTRIBUTE="+ str(kwargs.get("field_name"))])
    band.FlushCache()
    
def get_sr(layer):
    '''
    get the spatial reference of gdal.Dataset
    '''
    sr = osr.SpatialReference(str(layer.GetSpatialRef()))
    # auto-detect epsg
    sr.AutoIdentifyEPSG()
    sr.ImportFromEPSG(int(sr.GetAuthorityCode(None)))
    return sr

def modify_attribute(input_shp_name):
    '''resolving the conflicting attributes and connect the classes with number'''
    # read the shp file and load layer and features
    drv =ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = drv.Open(input_shp_name, update=True)
    shp_layer = shp_ds.GetLayer()
    features = shp_layer.GetLayerDefn()
    for i in range(features.GetFieldCount()):
        defn = features.GetFieldDefn(i)
        print(f'{defn.GetName()} ->  {defn.GetType()} -> {defn.GetWidth()}')
    
    # create a new field to store the confilicting object type
    # field = ogr.FieldDefn('CLASS', ogr.OFTString)
    # field.SetWidth(2)
    # shp_layer.CreateField(field)
    
    
    dict = {"Fels" : 1, "Fels locker" : 2, "Felsbloecke" : 3, "Felsbloecke locker" : 4, 
            "Feuchtgebiet" : 5, "Fliessgewaesser" : 6, "Gebueschwald" : 7, "Gehoelzflaeche" : 8, "Gletscher" : 9, 
            "Lockergestein" : 10, "Lockergestein locker" : 11, "Schneefeld Toteis" : 12, "Stehende Gewaesser" : 13,
            "Wald" : 14, "Wald offen" : 15} 
    
    for feature in shp_layer:
        #'OBJEKTART' - WEST, 'OBJEKTART_'' - east
        if feature.GetField('OBJEKTART') == None:
            category = feature.GetField('OBJEKTART_')
        elif feature.GetField('OBJEKTART_') == None:
            category = feature.GetField('OBJEKTART')
        if feature.GetField('OBJEKTART') != None and feature.GetField('OBJEKTART_') != None:
            category = feature.GetField('OBJEKTART')
        number = dict[category]
        feature.SetField('CLASS', number)
        shp_layer.SetFeature(feature)
    shp_ds = None
    
if __name__ == '__main__':
    input_shp_name = "/data/xiaolong/mask_shp/Label.shp"
    output_raster_name = "/data/xiaolong/mask_shp/Label.tif"
    # modify_attribute(input_shp_name)
    vector2raster(input_shp_name, output_raster_name, 0.5, field_name = "CLASS")