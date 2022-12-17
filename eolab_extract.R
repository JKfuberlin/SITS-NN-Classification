library(exactextractr)
library(terra)
library(sf)
library(tictoc)
library(dplyr)

# codede

aoi.tiles = c("TLT", "TMT", "TNT", "ULU", "UMU", "UMV", "UNA", "UNU", "UNV")

for (g in aoi.tiles) {
path = '/codede/Sentinel-2/MSI/L2A-FORCE/'
s2dirs = list.dirs(path = path, recursive = T)
s2dirs.aoi = grep(pattern = g, s2dirs, value = T)

raster.paths = c()

# get tifs from each folder
for (i in s2dirs.aoi) {
  a = list.files(path = i, pattern = '*BOA_B*', full.names = T)
  raster.paths = append(raster.paths, a)
}
# todo: only keep months 4-9 using a regex

# raster.paths2 = dplyr::nth(raster.paths, 10)
datacube = terra::rast(raster.paths)
names(datacube) = raster.paths
shapes = sf::st_read('/home/eouser/shapes/fe_points_grid.gpkg')
shapes = dplyr::filter(shapes, Name == paste0('32', g))

# tic()
shapes.extract = terra::extract(datacube, shapes) 
# shapes.extract2 = sf::st_set_geometry(shapes.extract, NULL)
# toc()

write.csv(shapes.extract, file = paste0('/home/eouser/csv/points',g,'.csv'))
}

# shapes.extract = exactextractr::extract_extract(datacube, shapes)
# doesn't work with terras spatrasters. need to find another way for polygon-wise weighted extractions

poly.df = tibble(x = c(390000, 390050, 390050, 390000, 390000), y = c(5200050, 5200050, 5200000, 5200000, 5200050))
poly.shape = sf::st_as_sf(poly.df, coords = c('x', 'y')) %>% st_set_crs(32632)
datacube2 = raster::stack(datacube)
test = raster::crop(datacube2, poly.shape)

# inspect
ULU = read.csv('/home/j/FF/csv/ULU.csv')
names(which(colSums(is.na(ULU)) > 0))
