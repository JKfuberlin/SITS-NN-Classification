#library(exactextractr)
library(terra)
library(sf)
library(tictoc)
library(dplyr)
library(stringr)


# epsg code needs to be 3035 for shapes
tt = sf::st_read('/home/eouser/training10m_3035.gpkg')
tiles = sf::st_read('/home/eouser/shapes/force_grid.gpkg')
		    
tic()
for (g in tiles$Tile_ID)
g = as.character(g)
path = '/force/FORCE/C1/L2/ard'
s2dirs = list.dirs(path = path, recursive = T)
s2dirs.aoi = grep(pattern = g, s2dirs, value = T)

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

datacube = terra::rast(raster.paths)

# rename raster names to include time
names(datacube) = raster.paths2

shapes = sf::st_read('/home/eouser/training10m_3035.gpkg')

a = sf::st_bbox(datacube)
b = as(raster::extent(a[[1]],a[[3]],a[[2]],a[[4]]), "SpatialPolygons")
c = sf::st_as_sf(b)
sf::st_crs(c) = 3035
d = shapes[c,] # this is the subset of shapes within the extent of the datacube raster
print(nrow(d))

print("extracting")
tic()
shapes.extract = terra::extract(datacube, d, xy = T) 
#
#
# @Dongshen: This is where you have to change settings for extraction from polygons
# you might want to get the exactextractr package to work which is usually faster
# and lets you set different functions.

toc()
print("extraction done")

# now we need to join the label on the extracted values
# and assign an ID to each observation
# need to convert shapes to data.frame without geometry to be able to join

shapes.extract = as_tibble(shapes_extract)
lookup = as_tibble(shapes)
shapes.extract = tibble::rowid_to_column(shapes.extract, "ID")
shapes.extract$species = dplyr::left_join(shapes.extract, shapes$BST1_BA_1, by=c('ID'='fid_2'))

print(paste0("writing",g))
write.csv(shapes.extract2, file = paste0('/home/eouser/csv/points',g,'.csv'))
print(paste0(g,"written"))

toc()
