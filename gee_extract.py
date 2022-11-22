# imports
import ee
import os
import time
import random

# Authent and Initialize
ee.Authenticate()
ee.Initialize()

# explanation for settings:
# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
CLOUD_FILTER = 30
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.2
CLD_PRJ_DIST = 1.2
BUFFER = 50

# load shape
sdate = '2017-07-09' # incomplete scene is from july 11th
edate = '2022-07-25' # complete scene for comparison from 13th


### cloud masking part for Sentinel-2 (all adapted from Pia Labenski; be aware that this code is NOT PUBLIC
### (although mainly adpoted from GEE tutorial) -> do not share)
# https://colab.research.google.com/drive/1Ms8F2Gk3rX3TUR39_ERaL6jCXTBf2Fpa?usp=sharing#scrollTo=syPE_kJSSAd1
def get_s2_sr_cld_col(aoi, start_date, end_date):  # aoi, start_date, end_date
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')  # S2_SR
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))
    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    })), s2_sr_col


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)
    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));
    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))
    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)
    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)
    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER * 2 / 20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))
    # Add the final cloud-shadow mask to the image.
    return img.addBands(is_cld_shdw)  # replace with img_cloud_shadow.addBands(is_cld_shdw)


def maskClouds_s2(img):
    return img.updateMask(img.select('cloudmask').eq(0))


def get_date(img):
    date = ee.Date(img.get('system:time_start')).format('YYYYMMdd')
    img = img.set('date', date)
    return img

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num


# add delay to for-loop:
DELAY = True
STEPS = 1500
DELAYTIME = 20000  # e.g. 32,400 seconds == 9 hours

aoi = ee.FeatureCollection('users/jk90fub/somepoints')

for i in range(0, 1):
# for i in range(0, fc.size().getInfo()):
    feature = ee.Feature(aoi.toList(aoi.size()).get(i))
    s2 = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(feature).filterDate(sdate, edate)
    IC_list = s2.toList(1000) # creates list of first 10 entries of the image collection, need to pass value or function does not work
    IC_list_properties = IC_list.getInfo() # creates list holding properties of these images
    list_of_coords = []
    list_of_scenes = [] #create empty list with scene IDs to later subset s2 object
    for i in range(0,len(IC_list_properties)): #iterate over all scenes found for the plot
        dict_properties = IC_list_properties[i] # apparently this is a dict inside a dict
        dict_properties2 = dict_properties.get('properties')  # now we can get values by key
        scene_geom = dict_properties2.get('system:footprint')
        coords = scene_geom.get('coordinates')
        flat_list = []
        for sublist in coords:
            for item in sublist:
                flat_list.append(item)
        # upper_left = coords[0]
        # origin = upper_left[0]
        list_of_coords.append(tuple(flat_list)) # list_of_coords is a list of lists, needs to be turned into a tuple
    target_coords = (most_frequent(list_of_coords))
    for i in range(0,len(IC_list_properties)): #iterate over all scenes found for the plot
        dict_properties = IC_list_properties[i] # apparently this is a dict inside a dict
        dict_properties2 = dict_properties.get('properties')  # now we can get values by key
        dict_id = dict_properties.get('id')
        scene_geom = dict_properties2.get('system:footprint')
        coords = scene_geom.get('coordinates')
        flat_list = []
        for sublist in coords:
            for item in sublist:
                flat_list.append(item)
        flat_list = tuple(flat_list)
        if flat_list == target_coords: # if the coordinates of this scene == coords of a complete scene
            dict_properties3 = dict_properties.get('features')
            list_of_scenes.append(dict_id)
    eelist = ee.List(list_of_scenes)
    images = eelist.map(ee.Image())
    collection = ee.ImageCollection(eelist)
    s2 = collection.filterBounds(aoi).filterDate(sdate, edate).filter(ee.Filter.calendarRange(4,9,'month'))
    # now we only want scenes from the list
    for i in range(0, 3):
    # for i in range(0, fc.size().getInfo()):
        feature = ee.Feature(aoi.toList(aoi.size()).get(i))
        s2 = ee.ImageCollection('COPERNICUS/S2_SR')
        startDate = '2017-01-01'
        endDate = '2022-10-31'
        s2_sr_cld_col_eval, s2_sr_col = get_s2_sr_cld_col(feature.geometry(), startDate, endDate)
        s2_sr_cld_col_eval = s2_sr_cld_col_eval.map(get_date)
        s2_sr_cld_col_eval_disp = s2_sr_cld_col_eval.map(add_cld_shdw_mask)
        s2 = s2_sr_cld_col_eval_disp.map(maskClouds_s2)
        s2 = s2.map(lambda image: image.set({"spacecraft_id": image.get("SPACECRAFT_NAME")}))
        data = s2.map(lambda image:
                      image.reduceRegions(
                          collection=feature,
                          reducer=ee.Reducer.mean(),
                          # crs='EPSG:5070',
                          scale=10
                      )
                      .map(lambda feat:
                           feat.copyProperties(image, image.propertyNames()))).flatten()
        print(f'saving polygon {i}')
        ee.batch.Export.table.toDrive(
            collection=data,
            folder='extract',
            description=os.path.join('plot_' + str(i)),
            selectors=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'date', 'spacecraft_id','id'],
            fileFormat='CSV').start()
        print("Submit all tasks to Google Earth Engine")
        # GEE can only handle 3000 tasks at once -> add sleep time to avoid overflow
        # if DELAY:
        #     if (int(i) % int(STEPS)) == 0: # every STEPS steps of for-loop
        #         time.sleep(DELAYTIME)