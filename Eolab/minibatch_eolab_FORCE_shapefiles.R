library(sf)
library(tidyverse)
library(dplyr)

grid = sf::st_read('/home/j/Nextcloud/Shapes/FORCE_datacube_grid_DEU.gpkg')

for (square in 1:nrow(grid)) { # for every square in the grid
  this_square = grid[square,] # select
  this_ID = this_square$Tile_ID # get tile_ID
  bbox = st_bbox(this_square) # get bounding box dimension
  smaller_grid <- st_make_grid(bbox, cellsize = c(5000, 5000), what = "polygons")
  
  number = 0 # iterator
  result = data.frame()
  for (minibatch in 1:length(smaller_grid)) {
    this_batch = smaller_grid[minibatch]
    this_dir = paste0('/home/j/Nextcloud/Shapes/FORCE_minibatches/',this_ID, '/')
    if (!dir.exists(this_dir)){
      dir.create(this_dir)
    }else{
    }
    sf::st_write(this_batch, paste0(this_dir, number,'.gpkg'))
    number = number+1
  }
}
