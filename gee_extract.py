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
CLOUD_FILTER = 10
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.2
CLD_PRJ_DIST = 1.2
BUFFER = 50

# load shape
fc = ee.FeatureCollection('users/dongshew96/bw_polygons_pure')

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

# add delay to for-loop:
DELAY = False
STEPS = 3000
DELAYTIME = 20000 # e.g. 32,400 seconds == 9 hours

for i in range(25000, 26000):
# for i in range(0, fc.size().getInfo()):
    feature = ee.Feature(fc.toList(fc.size()).get(i))
    # info = fc.toList(fc.size()).get(i).getInfo()
    # poly_id = info['properties']['id']
    s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    startDate = '2017-01-01'
    endDate = '2021-12-31'
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
        folder=f'bw_polygons_pure_cloud{CLOUD_FILTER}',
        description=os.path.join('plot_' + str(i)),
        selectors=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'date', 'spacecraft_id','id'],
        fileFormat='CSV').start()
    # GEE can only handle 3000 tasks at once -> add sleep time to avoid overflow
    if DELAY:
        if (int(i + 1) % int(STEPS)) == 0: # every STEPS steps of for-loop
            time.sleep(DELAYTIME)