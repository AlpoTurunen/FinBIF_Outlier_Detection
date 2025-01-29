import geopandas as gpd
from shapely.strtree import STRtree

# Load the lakes and grid squares GeoPackages
lakes = gpd.read_file("C:/Users/alpoturu/projektit/anomality_detection/vesistot.gpkg")  # Replace with the path where you have water area data
grid = gpd.read_file("data_obs/YKJ_env_no_birds.gpkg") 
print("Files readed")

# Ensure both layers have the same CRS
if lakes.crs != grid.crs:
    lakes = lakes.to_crs(grid.crs)

lakes = lakes.make_valid()

# Initialize a new column in the grid for coastline lengths
grid["coastline_length"] = 0.0

# Create a spatial index for the lakes
lake_geometries = lakes.geometry.values
lake_index = STRtree(lake_geometries)
print("index created")

# Calculate coastline lengths for each grid square
for index, square in grid.iterrows():
    # Find lakes intersecting the current grid square using spatial index
    possible_matches_index = lake_index.query(square.geometry)
    possible_matches = gpd.GeoSeries([lake_geometries[i] for i in possible_matches_index])
    
    # Calculate shared boundaries
    shared_boundaries = possible_matches.intersection(square.geometry)
    
    # Sum up the lengths of shared boundaries
    total_length = sum(geom.length for geom in shared_boundaries if not geom.is_empty)
    
    # Assign the total length to the grid square
    grid.at[index, "coastline_length"] = total_length

# Save the result to a new file
grid.to_file("YKJ10km_polygon/YKJ_Corine_birds2_coastline", driver="GPKG")
