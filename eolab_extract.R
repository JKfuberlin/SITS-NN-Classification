# library(exactextractr)
library(terra)
library(sf)
library(tictoc)
library(dplyr)
library(stringr)


# epsg code needs to be 3035 for shapes

# prepare file structure
# tiles = c("TLT", "TMT", "TNT", "ULU", "UMU", "UMV", "UNA", "UNU", "UNV")
tt = st_read('/home/eouser/shapes/target_tiles_32632.gpkg')
tiles = tt$Tile_ID

tic()
for (g in tiles) {
path = '/force/FORCE/C1/L2/ard'
# path = '/codede/Sentinel-2/MSI/L2A-FORCE/' 
s2dirs = list.dirs(path = path, recursive = T)
s2dirs.aoi = grep(pattern = g, s2dirs, value = T)
# s2dirs.aoi2 = stringr::str_subset(
#   s2dirs.aoi, 
#   pattern = "/codede/Sentinel-2/MSI/L2A-FORCE//20\\d+/0([4-9])/\\d+."
#   )
# s2dirs.aoi3 = stringr::str_subset(
#   s2dirs.aoi, 
#   pattern = "/codede/Sentinel-2/MSI/L2A-FORCE//20\\d+/10/\\d+."
# )
# s2dirs.aoi = c(s2dirs.aoi2, s2dirs.aoi3)

raster.paths = c()
# 20221127_LEVEL2_SEN2A_BOA.tif
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

# raster.paths2 = dplyr::nth(raster.paths, 10)

datacube = terra::rast(raster.paths)

# rename raster names to include time
names(datacube) = raster.paths2


shapes = sf::st_read('/home/eouser/training10m_3035.gpkg')


a = sf::st_bbox(datacube)
b = as(raster::extent(a[[1]],a[[3]],a[[2]],a[[4]]), "SpatialPolygons")
c = sf::st_as_sf(b)
sf::st_crs(c) = 3035
d = shapes[c,]
print(nrow(d))

# shapes = dplyr::filter(shapes, Name == paste0('32', g))

print("extracting")
tic()
shapes.extract = terra::extract(datacube, d, xy = T) 
#
#
# @Dongshen: This is where you have to change settings for extraction from polygons
# you might want to get the exactextractr package to work which is usually faster
# and lets you set different functions.



# shapes.extract2 = sf::st_set_geometry(shapes.extract, NULL)
toc()
print("extraction done")

shapes.extract2 = cbind(d, shapes.extract)
# shapes.extract2 = cbind(shapes2, shapes.extract)
# shapes.extract2 = dplyr::left_join(shapes, shape.extract, by = "fid_2")

print(paste0("writing",g))
write.csv(shapes.extract2, file = paste0('/home/eouser/csv/points',g,'.csv'))
print(paste0(g,"written"))
}
toc()


# shapes.extract = exactextractr::extract_extract(datacube, shapes)
# doesn't work with terras spatrasters. need to find another way for polygon-wise weighted extractions

poly.df = tibble(x = c(390000, 390050, 390050, 390000, 390000), y = c(5200050, 5200050, 5200000, 5200000, 5200050))
poly.shape = sf::st_as_sf(poly.df, coords = c('x', 'y')) %>% st_set_crs(32632)
datacube2 = raster::stack(datacube)
test = raster::crop(datacube2, poly.shape)

# inspect
ULU = read.csv('/home/j/FF/csv/ULU.csv')
names(which(colSums(is.na(ULU)) > 0))
