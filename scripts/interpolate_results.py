import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os, glob

data_folder = "data_results/geopackages_YKJ3"
shape_path = "data_env/SuomenValtakunta_2024_100k.shp"

shape = gpd.read_file(shape_path)

def Euclidean(x1,x2,y1,y2):
  return ((x1-x2)**2+(y1-y2)**2)**0.5

def IDW(data, LAT, LON, betta=2):
  array = np.empty((LAT.shape[0], LON.shape[0]))

  for i, lat in enumerate(LAT):
    for j, lon in enumerate(LON):
      weights = data.apply(lambda row: Euclidean(row.LONGITUDE, lon, row.LATITUDE, lat)**(-betta), axis = 1)
      z = sum(weights*data.probability)/weights.sum()
      array[i,j] = z
  return array

# Define LAT and LON grid for interpolation
lon_min, lat_min, lon_max, lat_max = shape.total_bounds
LON = np.linspace(lon_min, lon_max, 100)
LAT = np.linspace(lat_min, lat_max, 100)

# Process all GeoPackages in the folder
for file_path in glob.glob(os.path.join(data_folder, "*.gpkg")):
    print(f"Processing file: {file_path}")

    species_name = file_path.split("/")[-1].split(".")[0]

    # Load GeoPackage
    gdf = gpd.read_file(file_path)

    gdf['LONGITUDE'] = gdf.centroid.x
    gdf['LATITUDE'] = gdf.centroid.y

    # Perform interpolation
    result = IDW(gdf, LAT, LON)

    # Create a mask from the shape
    grid_lon, grid_lat = np.meshgrid(LON, LAT)  # Create the grid
    grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())]  # Convert to Points
    mask = np.array([shape.contains(point).any() for point in grid_points])  # Check if points are inside the shape
    mask = mask.reshape(grid_lon.shape)  # Reshape to match the grid

    # Apply the mask
    result[~mask] = np.nan  # Set values outside the shape to NaN

    # Plot interpolated results
    plt.figure(figsize=(10, 8))
    plt.imshow(result, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Interpolated probabilities for alli')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save the plot
    output_plot_path = os.path.join(data_folder, f"{species_name}_interpolated.png")
    plt.savefig(output_plot_path)
    plt.close()

    print(f"Saved interpolated plot to: {output_plot_path}")

print("done")
