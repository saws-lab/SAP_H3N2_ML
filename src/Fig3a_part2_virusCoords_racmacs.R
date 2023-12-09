library(Racmacs)

no_opt <- 5000

my_file <- file.path("..", "data", "titre_matrix.csv")
titer_table <- read.titerTable(my_file)

# Create the acmap object, specifying the titer table
map <- acmap(
  titer_table = titer_table
)

# Perform some optimization runs on the map object to try and determine a best map
map <- optimizeMap(
  map                     = map,
  number_of_dimensions    = 2,
  number_of_optimizations = no_opt,
  minimum_column_basis    = "none",
  options = list(ignore_disconnected = TRUE)
)

# keep the best map and its stress value
map <- keepBestOptimization(map)
stress <- allMapStresses(map)

plot(map)

# map <- rotateMap(map, 90)
# map <- reflectMap(map, axis = "x")

save.coords(
  map,
  "../data/titre_matrix_racmacs_coords.csv"
)
