# balancing training dataset for Transformer multilabel
library(tidyverse)
library(dplyr)
library(sf)
library(data.table)
library(plyr)

# load each csv and read species, as well as distribution only from large ass csv files
FE = sf::st_read('/home/j/Nextcloud/Shapes/mock/mock_fe.gpkg')
files = list.files('/home/j/data/polygons_object_test/', full.names =T)
csv = data.frame()
for (file in files) {
  a = data.table::fread(file=file, select = c("OBJECTID" , "BST1_BA_1", "BST1_BA_2","BST1_BA_3", "BST1_BAA_1", "BST1_BAA_2", "BST1_BAA_3" ))
  csv = rbind(csv,a)
}
save = csv
dictionary = read.csv('/home/j/Nextcloud/Baumartenschlüssel_FE.csv')
dict = dplyr::select(dictionary, "BST1_BA_1", "Multilabel_Encoding")
dict2 = as.list(dict)
dict1 = dict2[1] # FE codes
dict2 = dict2[2] # my encoding

# i want to replace species codes by Encoded values based on the key-value pairs
for (i in 1:length(dict1[[1]])) {
  a = dict1[[1]][[i]] # selecting the nth value of the respective list
  b = dict2[[1]][[i]]
  csv$BST1_BA_1 = dplyr::case_match(csv$BST1_BA_1, a ~ b, .default = csv$BST1_BA_1)
  csv$BST1_BA_2 = dplyr::case_match(csv$BST1_BA_2, a ~ b, .default = csv$BST1_BA_2)
  csv$BST1_BA_3 = dplyr::case_match(csv$BST1_BA_3, a ~ b, .default = csv$BST1_BA_3)
}
save2 = csv

# if two or three species are the same, add them up
for (i in 1:nrow(csv)) {
  # 1 & 2 the same species
  if (csv$BST1_BA_1[[i]] == csv$BST1_BA_2[[i]] & csv$BST1_BA_1[[i]] != csv$BST1_BA_3[[i]]) { # 1 == 2 but 1 != 3
    csv$BST1_BAA_1[[i]] = csv$BST1_BAA_1[[i]] + csv$BST1_BAA_2[[i]] # amount of coverage for same class is added...
    csv$BST1_BAA_2[[i]] = 0 # ...and superfluous classes deleted...
    csv$BST1_BA_2[[i]] = 0 # ...as well as their coverage
    }
  # 1 & 3 the same species
  if (csv$BST1_BA_1[[i]] == csv$BST1_BA_3[[i]] & csv$BST1_BA_1[[i]] != csv$BST1_BA_2[[i]]) {
    csv$BST1_BAA_1[[i]] = csv$BST1_BAA_1[[i]] + csv$BST1_BAA_3[[i]]
    csv$BST1_BAA_3[[i]] = 0
    csv$BST1_BA_3[[i]] = 0
  }
  
  # 2 & 3 the same species
  if (csv$BST1_BA_2[[i]] == csv$BST1_BA_3[[i]]  & csv$BST1_BA_2[[i]] != csv$BST1_BA_1[[i]]) {
    csv$BST1_BAA_2[[i]] = csv$BST1_BAA_2[[i]] + csv$BST1_BAA_3[[i]]
    csv$BST1_BAA_3[[i]] = 0
    csv$BST1_BA_3[[i]] = 0
  }
  
  # all three the same species
  if (csv$BST1_BA_1[[i]] == csv$BST1_BA_2[[i]] & csv$BST1_BA_2[[i]] == csv$BST1_BA_3[[i]]) {
    csv$BST1_BAA_1[[i]] = csv$BST1_BAA_1[[i]] + csv$BST1_BAA_2[[i]] + csv$BST1_BAA_3[[i]]
    csv$BST1_BAA_2[[i]] = 0
    csv$BST1_BAA_3[[i]] = 0
    csv$BST1_BA_2[[i]] = 0
    csv$BST1_BA_3[[i]] = 0
  }
}

# now i need to add columns to the df representing the labels, depending on the number of target species
columnsToAdd = paste("label", 1:12,sep="")
csv[,columnsToAdd]<-NA # adding new columns
csv[is.na(csv)] <- 0 # replacing NA with 0
# if species amount is > 10, replace by 1, if > 50, replace by 2


# create labels.csv
multi_labels.csv