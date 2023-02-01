#library(exactextractr)
library(terra)
library(sf)
library(tictoc)
library(dplyr)
library(stringr)
library(varhandle)
library(zoo) # quick and dirty interpolation

clean = F
interpolate = F
replace0 = T

# epsg code needs to be 3035 for shapes
# note that these shapes have a NEW ID which is different from the one in Forsteinrichtung (FE)
shapes = sf::st_read('/home/eouser/shapes/FEpoints10m_3035.gpkg')
tiles = varhandle::unfactor(unique(shapes$Tile_ID))

# initialise path to s2 timeseries
s2dirs = list.dirs(path = '/force/FORCE/C1/L2/ard', recursive = T)
  
tic()
for (g in tiles){
g = as.character(g)
s2dirs.aoi = grep(pattern = g, s2dirs, value = T)

raster.paths = c()
# pattern for regex 20221127_LEVEL2_SEN2A_BOA.tif
# get tifs from each folder
for (i in s2dirs.aoi) {
  a = list.files(path = i, full.names = T)
  b = stringr::str_subset(a, pattern = ".SEN2(A|B)_BOA.tif")
  raster.paths = append(raster.paths, b)
}

raster.paths2 = c()

# this creates a vector for renaming the datacube in order.
# date of capture + bandnumber referring to
# "BLUE"     "GREEN"   
#  "RED"      "REDEDGE1" "REDEDGE2" "REDEDGE3" "BROADNIR" "NIR"     
#  "SWIR1"    "SWIR2" 
for (j in raster.paths) {
  for (k in 1:10) {
    c = paste0(j, k)
    raster.paths2 = c(raster.paths2, c)
  }
}

tic()
datacube = terra::rast(raster.paths)
print(paste0('loading rasters ',g,' took ',toc()))

# rename raster names to include time
names(datacube) = raster.paths2

a = sf::st_bbox(datacube)
b = as(raster::extent(a[[1]],a[[3]],a[[2]],a[[4]]), "SpatialPolygons")
c = sf::st_as_sf(b)
sf::st_crs(c) = 3035
d = shapes[c,] # this is the subset of shapes within the extent of the datacube raster
print(nrow(d))


# @Dongshen: This is where you have to change settings for extraction from polygons
# you might want to get the exactextractr package to work which is usually faster
# and lets you set different functions.
print("extracting")
tic()
shapes.extract = terra::extract(datacube, d) 
toc()
print("extraction done")

if (interpolate == T) {
  path = '/home/eouser/csv/interpolate/'
  
  for (i in 1:nrow(shapes.extract)) { # for each observation in the df
    t = dplyr::slice(shapes.extract, i)
    interpolated.observation = dplyr::tibble()
    for (i in 1:10) {
      # create new timeline of values per band
      thisband %>% dplyr::select(num_range(range = seq(2, last_col(), by = 10)))
      interpolated.observation = t # create copy of observation that gets filled with interpolated values
      interpolated.band = zoo::na.approx(thisband) # this is where the interpolation happens
      interpolated.observation %>% mutate(num_range(range = seq(2, last_col(), by = 10)),
                                          value2))
    }
  }

  
} else if (clean == T) {
  # h <- janitor::remove_empty(h, which = "cols")
  # get rid of NA
  h = as_tibble(shapes.extract)
  na.dates = c()
  # find NA rows
  for (j in 1:ncol(h)-1) {
    k = h[1,1+j] # inspect first band of each observation band
    if (is.na(k)) {}
    # if it is NA, find out the date
    filename = names(k)
    # substr the date from this template "/force/FORCE/C1/L2/ard/X0058_Y0056/20221127_LEVEL2_SEN2A_BOA.tif10"
    date = substr(filename, 36, 43) 
    na.dates = c(na.dates, date)# add to list of dates with NA
}
  na.dates = unique(na.dates)
  #problem: in many cases, all pixels are clouded/NA at some point
  
} else if (replace0 == T){ # replace all NA by 0 equivalent to padding in neural net
  shapes.extract <- shapes.extract %>% replace(is.na(.), 0)
  path = '/home/eouser/csv/replace0/'} 
else {
  print('NA values will be written!')
  path = '/home/eouser/csv/'
}


# write single csv for each observation

print(paste0("writing",g))
for (i in 1:nrow(shapes.extract)) {
  t = dplyr::slice(shapes.extract, i)
  write.csv(t, file = paste0(path,t$ID,'.csv'))
}
print(paste0(g,"written"))
}

toc()
