# imports
import ee
import os
import time

# Authent and Initialize
print('aeo')
ee.Authenticate()
print('aeo')
ee.Initialize()

CLOUD_FILTER = 90
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.2
CLD_PRJ_DIST = 1.2
BUFFER = 20

# load shape
fc = ee.FeatureCollection('users/jk90fub/somepoints')

def getQABits(image, start, end, newName):
      # Compute the bits we need to extract.
     pattern = 0
     for i in range(start, end + 1):
          pattern += 2 ** i
    # Return a single band image of the extracted QA bits, giving the band a new name.
     return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

    ## a method to mask out cloud shadows
def cloud_shadows(image):
   ## Select the QA band.
     QA = image.select(['pixel_qa'])
    ## Get the internal_cloud_algorithm_flag bit.
     return getQABits(QA, 3, 3, 'Cloud_shadows').eq(0)
     ## Return an image masking out cloudy areas.

## a method to mask out clouds
def clouds(image):
    ## Select the QA band.
    QA = image.select(['pixel_qa'])
    ## Get the internal_cloud_algorithm_flag bit.
    return getQABits(QA, 5, 5, 'Cloud').eq(0)
    ## Return an image masking out cloudy areas.
    ## a method to mask out snow

def snow(image):
   ## Select the QA band
   QA = image.select(['pixel_qa'])
   ## Get the internal_cloud_algorithm_flag bit.
   return getQABits(QA, 4, 4, 'Snow').eq(0)
   ## Return an image masking out cloudy areas.
   ## combined method that masks out snow, cloud_shadows and clouds

def maskClouds(image):
   s = snow(image)
   cs = cloud_shadows(image)
   c = clouds(image)
   image = image.updateMask(s)
   image = image.updateMask(cs)
   return image.updateMask(c)


### cloud masking part for Sentinel-2 (all adapted from Pia Labenski; be aware that this code is NOT PUBLIC
### (although mainly adpoted from GEE tutorial) -> do not share)
# https://colab.research.google.com/drive/1Ms8F2Gk3rX3TUR39_ERaL6jCXTBf2Fpa?usp=sharing#scrollTo=syPE_kJSSAd1
def get_s2_sr_cld_col(aoi, start_date, end_date):  # aoi, start_date, end_date
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')  # S2_SR
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    print('jusquici')
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))
    print('toutvabien')
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


def renameS2(img):
    return img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'])


def prepS2(img):
    img = add_cld_shdw_mask(img)
    img = maskClouds_s2(img).unmask(-9999)
    img = renameS2(img)
    return img


def prepS2(img):
    img = add_cld_shdw_mask(img)
    img = maskClouds_s2(img).unmask(-9999)
    img = renameS2(img)
    return img


def get_date(img):
    date = ee.Date(img.get('system:time_start')).format('YYYYMMdd')
    img = img.set('date', date)
    return img

feature = ee.Feature(fc.toList(fc.size()).get(1)) # first feature as example: get(0); toList(235) is max number of features for FVA dataset

s2 = ee.ImageCollection('COPERNICUS/S2_SR')
print(s2.first().getInfo())


######################################################
####### workflow: get landsat time series data #######
######################################################

## set startDate and endDate for the feature
# print(feature.get('date').getInfo())
# endDate = ee.Date(fc.get('date')).format('YYYY-MM-dd')
# print(endDate.getInfo())
# print(endDate.format('YYYY-MM-dd').getInfo())
# startDate = ee.Date(ee.Date(feature.get('date')).advance(-6, 'years')) # .format('YYYY-MM-dd') # 208 weeks earlier
# print(ee.Date(endDate).advance(-207, 'weeks').getInfo()) # startDate
# print(startDate.getInfo())

startDate = '2015-07-01'
endDate = '2022-10-30'
s2_sr_cld_col_eval, s2_sr_col = get_s2_sr_cld_col(feature.geometry(), startDate, endDate)
# print(s2_sr_cld_col_eval.first().getInfo())
# print(s2_sr_col.first().getInfo())



s2_sr_cld_col_eval = s2_sr_cld_col_eval.map(get_date)
# dates = s2_sr_cld_col_eval.aggregate_array('date')
# print(s2_sr_cld_col_eval.first().getInfo())

s2_sr_cld_col_eval_disp = s2_sr_cld_col_eval.map(add_cld_shdw_mask)
# print(s2_sr_cld_col_eval_disp.first().getInfo())

# Mask out cloudy pixels in original image
s2 = s2_sr_cld_col_eval_disp.map(maskClouds_s2)
# print(s2.first().getInfo())
s2 = s2.map(renameS2)
s2 = s2.map(lambda image: image.set({"spacecraft_id": image.get("SPACECRAFT_NAME")}))
# print(s2.first().getInfo())

## combined collection of all Landsat images
# s2 = ee.ImageCollection('COPERNICUS/S2_SR')

# s2 = ee.ImageCollection('COPERNICUS/S2_SR').map(prepS2)

# s2 = ee.ImageCollection('COPERNICUS/S2_SR')
# print(s2.first().getInfo())


data = s2.map(lambda image:
                       image.reduceRegions(
                           collection = feature,
                           reducer=ee.Reducer.mean(),
                           # crs='EPSG:5070',
                           scale=30
                       )
                  .map(lambda feat:
                       feat.copyProperties(image, image.propertyNames()))).flatten()


# print(data.first().propertyNames().getInfo())
# print(data.size().getInfo())

### prepare the export
plotnumber = feature.get('plotID').getInfo() # FVA data: 'plot'
datestring = feature.get('date').getInfo()
datestring = datestring.replace('-', '_')

### export
ee.batch.Export.table.toDrive(
    collection=data,
    folder='senf_tree_health_poly_plot_area_all_data',
    description=os.path.join('senf_timeseries_plot_' + str(plotnumber) + '_date_' + str(datestring)),
    selectors=['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'date', 'spacecraft_id', 'mort_0', 'mort_1',
               'mort_2', 'mort_3', 'mort_4', 'mort_5', 'mort_7', 'mort_8', 'mort_9', 'plot', 'area_ha', 'abswood', 'lostwoodpc', '.geo'],
    fileFormat='CSV').start()

