library(Racmacs)

#setwd(getSrcDirectory(function(){})[1])
my_file <- file.path("..", "data", "titer_matrix.csv")
titer_table <- read.titerTable(my_file)

# Create the acmap object, specifying the titer table
map <- acmap(
  titer_table = titer_table
)

# Perform some optimization runs on the map object to try and determine a best map
map <- optimizeMap(
  map                     = map,
  number_of_dimensions    = 2,
  number_of_optimizations = 1000,
  minimum_column_basis    = "none",
  options = list(ignore_disconnected = TRUE)
)

save.coords(
  map,
  "../results/Fig3a_seasonal_antigenic_cartography/titer_matrix_racmacs_coords.csv",
  optimization_number = 1000,
  antigens = TRUE,
  sera = TRUE
)
