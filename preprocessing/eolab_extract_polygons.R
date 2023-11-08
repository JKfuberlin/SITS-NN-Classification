# build list/ overview polygon of datacube boundaries
# read each polygon
# check which time series need to be loaded
# merge the time series in case of border polygons
# extract



#automatic install of packages if they are not installed already
# source: https://www.blasbenito.com/post/02_parallelizing_loops_with_r/
list.of.packages <- c(
  "tidyverse", # data handling
  "tictoc", # benchmarking
  "dplyr", # data handling
  "terra", # state of the art package for handling raster tif
  "sf",
  "doParallel",
  "foreach",
  "exactextractr" # extraction
)

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages) > 0){
  install.packages(new.packages, dep=TRUE)
}

#loading packages
for(package.i in list.of.packages){
  suppressPackageStartupMessages(
    library(
      package.i,
      character.only = TRUE
    )
  )
}

terra::gdalCache(37373) # setting GDAL cache to load raster blocks from datacube to speed up performance


# epsg code needs to be 3035 for shapes
# note that these shapes have a NEW ID which is different from the one in Forsteinrichtung (FE)
shapes = sf::st_read('/my_volume/object_based/shapes/cleaned_fe_polys.gkpg')
tiles = sf::st_read('/home/eouser/shapes/target_tiles.gpkg')
s2dirs = list.dirs(path = '/force/FORCE/C1/L2/ard', recursive = T) # initialize path to s2 time series
path = paste0('/my_volume/polygons2/')

# buffer
# shapes = sf::st_buffer(shapes, -10)

shapes = sf::st_transform(shapes, crs = 3035)
tiles = sf::st_transform(tiles, crs = 3035)

for (tile in 1:nrow(tiles)) {
  # looping over FORCE tiles
  this.tile = tiles[tile,]
  print('this tile... ')
  print(this.tile)
  df = shapes[1,]
  df = df[0,] # creating empty df that matches the stands
  shapes.subset = shapes[this.tile,] # simple subset selecting all FE polygons that intersect the tile
  within = sf::st_contains_properly(this.tile, shapes.subset, sparse = F) # list of subset that are contained properly
  # i think this is unnecessary at this moment because i did this subset before
  # i would love to do this using st_filter but i think it does not work
  # within2 = sf::st_filter(this.tile, shapes.subset, .predicate = st_contains_properly)
  print('finding target stands in tile')
  target.stands = list() # now i want to retrieve the stands that match the ones in within
  for (trues in 1:length(within[1,])) { # by checking whether the output of within is true or not
    this = within[trues]
    if (this == "TRUE") {
      target.stands= append(target.stands, trues) # this should now contain the indices of target stands
    }
  }
  target.stands = unlist(target.stands) # turn list of  stands into a vector, so %in% can be applied to subset shapes

  for (stand in 1:nrow(shapes.subset)) {
    if (stand %in% target.stands){ # only if the index of the stand is within the target ones...
      df = df %>% tibble::add_row(shapes.subset[stand,]) # ... i add it to the df
    }
  }
  shapes.extract = df # these shapes will be used for extraction
  g = this.tile$Tile_ID
  s2dirs.aoi = grep(pattern = g, s2dirs, value = T)

  raster.paths = c()
  # pattern for regex 20221127_LEVEL2_SEN2A_BOA.tif
  # get tifs from each folder
  print('getting tifs from each folder')
  for (s2 in s2dirs.aoi) {
    a = list.files(path = s2, full.names = T)
    b = stringr::str_subset(a, pattern = ".SEN2(A|B)_BOA.tif")
    raster.paths = append(raster.paths, b)
  }

  raster.paths2 = c()

  # this creates a vector for renaming the datacube in order.
  # date of capture + bandnumber referring to
  # "BLUE"     "GREEN"
  #  "RED"      "REDEDGE1" "REDEDGE2" "REDEDGE3" "BROADNIR" "NIR"
  #  "SWIR1"    "SWIR2"
  for (path in raster.paths) {
    for (band in 1:10) {
      c = paste0(path, band)
      raster.paths2 = c(raster.paths2, c)
    }
  }

  print('stacking datacube')
  tic()
  # datacube = raster::stack(raster.paths)
  datacube = terra::rast(raster.paths)
  toc()
  print(paste0('loading rasters ',g))

  # rename raster names to include time
  names(datacube) = raster.paths2

  print("extracting")

  mean = exactextractr::exact_extract(datacube, shapes.extract, fun = 'mean')
  shapes.extract$mean <- mean

  max = exactextractr::exact_extract(datacube, shapes.extract, fun = 'max')
  shapes.extract$max <- max

  min = exactextractr::exact_extract(datacube, shapes.extract, fun = 'min')
  shapes.extract$min <- min

  variance = exactextractr::exact_extract(datacube, shapes.extract, fun = 'variance')
  shapes.extract$variance <- variance

  # stdev = exactextractr::exact_extract(datacube, shapes.extract, fun = 'stdev')
  # shapes.extract$stdev <- stdev # can be calculated from variance if needed

  q05 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 0.05)
  shapes.extract$q05 <- q05
  q25 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 0.25)
  shapes.extract$q25 <- q25
  median = exactextractr::exact_extract(datacube, shapes.extract, fun = 'median')
  shapes.extract$median <- median
  q75 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 0.75)
  shapes.extract$q75 <- q75
  q95 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 0.95)
  shapes.extract$q95 <- q95


print("extraction done")
path = paste0('/my_volume/polygons2/')
print("writing")

for (pixel in 1:nrow(shapes.extract)) { # for each observation in the df
  t = dplyr::slice(shapes.extract, pixel)
  t = sf::st_drop_geometry(t)
  # save the row as csv
  # print(paste0(path,t$OBJECTID,'.csv'))
  write.csv(t, file = paste0(path,t$OBJECTID,'.csv'))
}
gc()
}

# debug:

for (tile in 1:nrow(tiles)) {
  # looping over FORCE tiles
  this.tile = tiles[tile,]
  print('this tile... ')
  print(this.tile)
  df = shapes[1,]
  df = df[0,] # creating empty df that matches the stands
  shapes.subset = shapes[this.tile,] # simple subset selecting all FE polygons that intersect the tile
  within = sf::st_contains_properly(this.tile, shapes.subset, sparse = F) # list of subset that are contained properly
  # i think this is unnecessary at this moment because i did this subset before
  # i would love to do this using st_filter but i think it does not work
  # within2 = sf::st_filter(this.tile, shapes.subset, .predicate = st_contains_properly)
  print('finding target stands in tile')
  target.stands = list() # now i want to retrieve the stands that match the ones in within
  for (trues in 1:length(within[1,])) { # by checking whether the output of within is true or not
    this = within[trues]
    if (this == "TRUE") {
      target.stands= append(target.stands, trues) # this should now contain the indices of target stands
    }
  }
  target.stands = unlist(target.stands) # turn list of  stands into a vector, so %in% can be applied to subset shapes

  for (stand in 1:nrow(shapes.subset)) {
    if (stand %in% target.stands){ # only if the index of the stand is within the target ones...
      df = df %>% tibble::add_row(shapes.subset[stand,]) # ... i add it to the df
    }
  }
  shapes.extract = df # these shapes will be used for extraction
  g = this.tile$Tile_ID
  s2dirs.aoi = grep(pattern = g, s2dirs, value = T)

  raster.paths = c()
  # pattern for regex 20221127_LEVEL2_SEN2A_BOA.tif
  # get tifs from each folder
  print('getting tifs from each folder')
  for (s2 in s2dirs.aoi) {
    a = list.files(path = s2, full.names = T)
    b = stringr::str_subset(a, pattern = ".SEN2(A|B)_BOA.tif")
    raster.paths = append(raster.paths, b)
  }

  raster.paths2 = c()

  # this creates a vector for renaming the datacube in order.
  # date of capture + bandnumber referring to
  # "BLUE"     "GREEN"
  #  "RED"      "REDEDGE1" "REDEDGE2" "REDEDGE3" "BROADNIR" "NIR"
  #  "SWIR1"    "SWIR2"
  for (path in raster.paths) {
    for (band in 1:10) {
      c = paste0(path, band)
      raster.paths2 = c(raster.paths2, c)
    }
  }
  print('stacking datacube')
  tic()
  # datacube = raster::stack(raster.paths)
  datacube = terra::rast(raster.paths)
  toc()
  print(paste0('loading rasters ',g))
  # rename raster names to include time
  names(datacube) = raster.paths2
  print("extracting")
  print("extraction done")
  path = paste0('/my_volume/polygons2/')
  print("writing")
  gc()
}

tile=1
this.tile = tiles[tile,]
df = shapes[1,]
df = df[0,] # creating empty df that matches the stands
shapes.subset = shapes[this.tile,] # simple subset selecting all shapes that intersect the tile
within = sf::st_contains_properly(this.tile, shapes.subset, sparse = F) # list of subset that are contained properly
# i would love to do this using st_filter but i think it does not work
# within2 = sf::st_filter(this.tile, shapes.subset, .predicate = st_contains_properly)

target.stands = list() # now i want to retrieve the stands that match the ones in within
for (trues in 1:length(within[1,])) { # by checking whether the output of within is true or not
  this = within[trues]
  if (this == "TRUE") {
    target.stands= append(target.stands, trues) # this should now contain the indices of target stands
  }
}
target.stands = unlist(target.stands) # turn into a vector, so %in% can be applied to subset shapes

for (stand in 1:nrow(shapes.subset)) {
  if (stand %in% target.stands){ # only if the index of the stand is within the target ones...
    df = df %>% tibble::add_row(shapes.subset[stand,]) # ... i add it to the df
  }
}
shapes.extract = df
g = this.tile$Tile_ID
s2dirs.aoi = grep(pattern = g, s2dirs, value = T)

raster.paths = c()
# pattern for regex 20221127_LEVEL2_SEN2A_BOA.tif
# get tifs from each folder
for (s2 in s2dirs.aoi) {
  a = list.files(path = s2, full.names = T)
  b = stringr::str_subset(a, pattern = ".SEN2(A|B)_BOA.tif")
  raster.paths = append(raster.paths, b)
}

raster.paths2 = c()

# this creates a vector for renaming the datacube in order.
# date of capture + bandnumber referring to
# "BLUE"     "GREEN"
#  "RED"      "REDEDGE1" "REDEDGE2" "REDEDGE3" "BROADNIR" "NIR"
#  "SWIR1"    "SWIR2"
for (path in raster.paths) {
  for (band in 1:10) {
    c = paste0(path, band)
    raster.paths2 = c(raster.paths2, c)
  }
}

tic()
# datacube = raster::stack(raster.paths)
datacube = terra::rast(raster.paths)
toc()
print(paste0('loading rasters ',g))

# rename raster names to include time
names(datacube) = raster.paths2

print("extracting")

mean = exactextractr::exact_extract(datacube, shapes.extract, fun = 'mean')
shapes.extract$mean <- mean

max = exactextractr::exact_extract(datacube, shapes.extract, fun = 'max')
shapes.extract$max <- max

min = exactextractr::exact_extract(datacube, shapes.extract, fun = 'min')
shapes.extract$min <- min



variance = exactextractr::exact_extract(datacube, shapes.extract, fun = 'variance')
shapes.extract$variance <- variance

# stdev = exactextractr::exact_extract(datacube, shapes.extract, fun = 'stdev')
# shapes.extract$stdev <- stdev # can be calculated from variance if needed

q05 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 05)
shapes.extract$q75 <- q05
q25 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 25)
shapes.extract$q25 <- q25
median = exactextractr::exact_extract(datacube, shapes.extract, fun = 'median')
shapes.extract$median <- median
q75 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 75)
shapes.extract$q75 <- q75
q95 = exactextractr::exact_extract(datacube, shapes.extract, fun = 'quantile', quantiles = 95)
shapes.extract$q75 <- q95



print("extraction done")
path = paste0('/my_volume/polygons2/')
print("writing")

for (pixel in 1:nrow(shapes.extract)) { # for each observation in the df
  t = dplyr::slice(shapes.extract, pixel)
  t = sf::st_drop_geometry(t)
  # save the row as csv
  print(paste0(path,t$OBJECTID,'.csv'))
  write.csv(t, file = paste0(path,t$OBJECTID,'.csv'))
}
gc()
done = append(done, t$ID)
gc()

