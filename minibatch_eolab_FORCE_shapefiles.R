# this scripts creates smaller subsets of the FORCE grid to enable processing on EOLAB
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
      # minibatch = sf::as_Spatial(minibatch, 'sf')
      number = number+1
      this_batch$number = number
      this_batch$Tile_ID = this_ID
      result = dplyr::add_row(result, this_batch)
    }
}

sf::st_write(minibatch, '/home/j/Nextcloud/Shapes/FORCE_minibatches.gpkg')

# the problem however, this is a single gpkg, now i need to write the inference script so it can read single squares from it.