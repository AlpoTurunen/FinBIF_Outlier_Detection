import numpy as np
import rasterio
from shapely.geometry import box
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from typing import Union, Tuple
from elapid.types import Vector
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point, LineString, MultiPoint, MultiLineString
from scipy.spatial import KDTree
from load_data import fetch_json_with_retry
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os, csv, rasterio, pyproj, warnings
from sklearn.model_selection import GridSearchCV
import seaborn as sns

warnings.filterwarnings('ignore')

# Helper function for Grid Search
def perform_grid_search(model, param_grid, X_train, y_train, weights_train, cv, scoring='roc_auc'):
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv,
        scoring=scoring, n_jobs=-1
    )
    grid_search.fit(X_train, y_train, sample_weight=weights_train)
    return grid_search.best_estimator_

def find_highly_correlated_features(X, threshold=0.9):
    # Compute correlation matrix
    corr_matrix = X.corr().abs()

    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Find highly correlated features (threshold > 0.8)
    high_corr_features = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                feature = corr_matrix.columns[i]
                high_corr_features.append(feature)

    print(f"Highly correlated features (correlation > {threshold}) to drop: {high_corr_features}")
    return high_corr_features

# explicit function to normalize array
def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def get_statistics(model, model_name, species_name, y_test, predictions, probs):
    """
    Print performance statistics for a model and save results to a CSV file.
    """
    
    # Extract hyperparameters
    model_params = model.get_params()
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probs) if probs is not None else None

    # Prepare row for saving
    result_row = {
        'Species': species_name,
        'Model': model_name,
        'Hyperparameters': model_params,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

    return result_row

def save_results_to_csv(csv_file, result_row):

    # Define the CSV file path
    file_exists = os.path.isfile(csv_file)
    
    # Save results to a CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['Species', 'Model', 'Hyperparameters', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the current result row
        writer.writerow(result_row)


def extract_raster_values_with_buffer(gdf, paths, buffer_distance=100):
    """
    Extract mean raster values within a buffer around points for single-band rasters, ignoring nodata values.

    Parameters:
    - point_gdf: GeoDataFrame containing the points.
    - paths: List of file paths to rasters.
    - buffer_distance: Buffer distance in the same units as the CRS of the rasters (default: 100).

    Returns:
    - GeoDataFrame with extracted mean raster values added as new columns.
    """
 
    gdf['buffer'] = gdf.geometry.buffer(buffer_distance)

    for path in paths:
        print(f"Processing raster: {path} with buffer of {buffer_distance} meters")

        with rasterio.open(path) as raster:
            # Read the single band (band 1)
            band = raster.read(1)
            nodata_value = raster.nodata  # Get the nodata value for the raster
            transform = raster.transform
            predictor_values = pd.DataFrame()
            means = []
            predictor_name = path.split('/')[-1].removesuffix('.tif')  # Raster name without path and .tif


            # Iterate over each point
            for _, row in gdf.iterrows():

                # Extract a buffer around the point
                buffer_geom = [row['buffer']]
                
                # Mask the raster with the buffer geometry
                out_image, _ = mask(raster, buffer_geom, crop=True)

                # Flatten the masked raster data to 1D array and remove nodata values
                out_image = out_image.flatten()
                out_image = out_image[out_image != nodata_value]

                 # If no valid data, skip to the next point
                if len(out_image) == 0:
                    continue

                # Calculate the mean number of pixels within the buffer
                mean_value = np.sum(out_image) / len(out_image)

                means.append(mean_value)

        # Add the mean values as a new column in the predictor_values DataFrame
        predictor_values[predictor_name] = pd.Series(means)

    # Remove the temporary buffer column
    gdf.drop(columns='buffer', inplace=True)

    # Concatenate the point GeoDataFrame with the predictor values
    return pd.concat([gdf, predictor_values], axis=1)

def extract_raster_values(point_gdf, paths):
    """
    Extract values at points for single-band rasters, ignoring nodata values.

    Parameters:
    - point_gdf: GeoDataFrame containing the points.
    - paths: List of file paths to rasters.

    Returns:
    - GeoDataFrame with extracted raster values added as new columns.
    """
    predictor_values = pd.DataFrame()

    for path in paths:
        print(f"Processing raster: {path}")

        values = []

        with rasterio.open(path) as raster:
            # Read the single band (band 1)
            band = raster.read(1)
            nodata_value = raster.nodata  # Get the nodata value for the raster

            # Iterate over each point
            for point in point_gdf.geometry:
                # Get the row and column corresponding to the point
                row, col = raster.index(point.x, point.y)
                
                # Check if the row/col is within the bounds of the raster
                if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
                    pixel_value = band[row, col]
                    # Check if the pixel value is equal to the nodata value
                    if pixel_value == nodata_value:
                        values.append(None)  # Ignore nodata values
                        continue
                    else:
                        values.append(pixel_value)
                        continue
                else:
                    values.append(None) # If the point is located outside of raster extent

        predictor_name = path.split('/')[-1].removesuffix('.tif') # raster name without path and .tif

        # Add the extracted values as a new column in the predictor_values DataFrame
        predictor_values[predictor_name] = pd.Series(values)

    # Concatenate the point GeoDataFrame with the predictor values
    return pd.concat([point_gdf, predictor_values], axis=1)

def remove_duplicates(gdf, grid_size=100):
    """
    Remove duplicate occurrence points by retaining only one point per 1 km pixel.
    
    :param gdf: GeoDataFrame containing the occurrence points
    :param grid_size: The size of the grid cells in meters (default is 1000m)
    :return: GeoDataFrame with duplicates removed
    """
    # Ensure the points are in a projected CRS for accurate distance-based operations
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3067)  # Reproject to a CRS with meters, e.g., EPSG:3857

    # Create a grid covering the study area
    grid_gdf = create_grid(gdf, grid_size=grid_size)

    # Spatial join between points and grid
    points_in_grid = gpd.sjoin(gdf, grid_gdf, how='left', predicate='within')

    # Remove duplicates by retaining only one point per grid cell
    points_deduplicated = points_in_grid.drop_duplicates(subset='index_right').reset_index(drop=True)

    print(f"In total {len(gdf)-len(points_deduplicated)} duplicates were removed with a grid size {grid_size}")

    return points_deduplicated

def create_grid(gdf, grid_size=1000):
    """
    Create a regular grid of square polygons (grid_size) that covers the extent of the GeoDataFrame.
    
    :param gdf: GeoDataFrame containing the points
    :param grid_size: The size of the grid cells in meters (default is 1000m)
    :return: GeoDataFrame with the grid polygons
    """
    # Get the bounds of the study area
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Create a grid of 1 km squares (in projected CRS, such as UTM or EPSG:3857)
    xmin, ymin, xmax, ymax = bounds
    grid_cells = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            grid_cells.append(box(x, y, x + grid_size, y + grid_size))
            y += grid_size
        x += grid_size
    
    # Create a GeoDataFrame for the grid
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf.crs)
    print(f"A grid of {grid_size} meter squares was created")
    return grid

def calculate_land_cover_proportions(gdf, raster_path, buffer_size=100):
    """
    Calculate the proportion of each land cover type within a 10-meter buffer
    around points in the GeoDataFrame based on CORINE raster data.

    Parameters:
    - gdf: GeoDataFrame with point geometries.
    - raster_path: Path to the CORINE raster file.

    Returns:
    - Updated GeoDataFrame with land cover proportion columns.
    """
    print("Calculating corine land cover proportions...")

    # Create a buffer of 10 meters around each point in the GeoDataFrame
    gdf['buffer'] = gdf.geometry.buffer(buffer_size)

    land_cover_classes= {1: 'Urban',
                         2: 'Park',
                         3: 'Rural',
                         4: 'Forest',
                         5: 'Open forest',
                         6: 'Fjell',
                         7: 'Open area',
                         8: 'Wetland',
                         9: 'Open bog',
                         10: 'Freshwater',
                         11: 'Marine',
                         }

    # Open the CORINE raster file
    with rasterio.open(raster_path) as src:
        # Initialize columns for each land cover class
        for class_id, class_name in land_cover_classes.items():
            gdf[class_name] = 0.0

        # Loop through each row (point) in the GeoDataFrame
        for idx, row in gdf.iterrows():
            # Extract the buffer geometry
            buffer_geom = [row['buffer']]

            # Mask the raster with the buffer geometry
            out_image, _ = mask(src, buffer_geom, crop=True)
            
            # Flatten the masked raster data to 1D array and remove nodata values
            out_image = out_image.flatten()
            out_image = out_image[out_image != src.nodata]

            # If no valid data, skip to the next point
            if len(out_image) == 0:
                continue

            # Calculate the total number of pixels within the buffer
            total_pixels = len(out_image)

            # Calculate the proportion of each land cover class
            for class_id, class_name in land_cover_classes.items():
                class_pixels = np.sum(out_image == class_id)
                proportion = class_pixels / total_pixels
                gdf.at[idx, class_name] = proportion

    # Remove the temporary buffer column
    gdf.drop(columns='buffer', inplace=True)

    return gdf

def normalise_columns(gdf):
    """
    Normalises all numerical columns in a (Geo)DataFrame to the range between 0 and 1.

    Parameters:
    - gdf: GeoDataFrame or DataFrame.

    Returns:
    - Updated DataFrame with rescaled values between 0-1.
    """
    # Iterate over all columns in the DataFrame
    for column in gdf.columns:
        # Check if the column contains numerical data types
        if pd.api.types.is_numeric_dtype(gdf[column]):
            min_val = gdf[column].min()
            max_val = gdf[column].max()
            range_val = max_val - min_val
            if range_val != 0:
                gdf[column] = (gdf[column] - min_val) / range_val
            else:
                gdf[column] = 0.0  # Set to 0 if all values are the same
    return gdf

def sample_polygon(polygon_path: str, count: int) -> gpd.GeoDataFrame:
    """Create a random geographic sample of points within a polygon's extent.

    Args:
        polygon_path: vector polygon shapefile path to sample locations from
        count: number of samples to generate

    Returns:
        points: Point geometry GeoDataFrame
    """

    print(f"Generating sample background points from the file {polygon_path}...")

    # Load the polygon shapefile
    polygons = gpd.read_file(polygon_path)
    bounds = polygons.total_bounds  # xmin, ymin, xmax, ymax
    xmin, ymin, xmax, ymax = bounds

    # Generate random points within the bounding box until we have the desired count
    points = []
    while len(points) < count:
        x_random = np.random.uniform(xmin, xmax, count * 2)
        y_random = np.random.uniform(ymin, ymax, count * 2)
        candidate_points = [Point(x, y) for x, y in zip(x_random, y_random)]

        # Filter points that fall within the polygon(s)
        valid_points = [pt for pt in candidate_points if polygons.contains(pt).any()]
        points.extend(valid_points)

    # Trim the list to the requested count
    points = points[:count]

    # Convert points to a GeoSeries
    crs = polygons.crs
    gdf = gpd.GeoDataFrame({'geometry': points}, crs=crs)
    
    print(f"In total {len(gdf)} background sample points were genearated")

    return gdf

def validate_gpd(geo: Vector) -> None:
    """Validates whether an input is a GeoDataFrame or a GeoSeries.

    Args:
        geo: an input variable that should be in GeoPandas format

    Raises:
        TypeError: geo is not a GeoPandas dataframe or series
    """
    if not (isinstance(geo, gpd.GeoDataFrame) or isinstance(geo, gpd.GeoSeries)):
        raise TypeError("Input must be a GeoDataFrame or GeoSeries")

def parse_crs_string(string: str) -> str:
    """Parses a string to determine the CRS/spatial projection format.

    Args:
        string: a string with CRS/projection data.

    Returns:
        crs_type: Str in ["wkt", "proj4", "epsg", "string"].
    """
    if "epsg:" in string.lower():
        return "epsg"
    elif "+proj" in string:
        return "proj4"
    elif "SPHEROID" in string:
        return "wkt"
    else:
        return "string"

def string_to_crs(string: str) -> rasterio.crs.CRS:
    """Converts a crs/projection string to a pyproj-readable CRS object

    Args:
        string: a crs/projection string.

    Returns:
        crs: the coordinate reference system
    """
    crs_type = parse_crs_string(string)

    if crs_type == "epsg":
        auth, code = string.split(":")
        crs = rasterio.crs.CRS.from_epsg(int(code))
    elif crs_type == "proj4":
        crs = rasterio.crs.CRS.from_proj4(string)
    elif crs_type == "wkt":
        crs = rasterio.crs.CRS.from_wkt(string)
    else:
        crs = rasterio.crs.CRS.from_string(string)

    return crs

def crs_match(crs1: Union[pyproj.CRS, str], crs2: Union[pyproj.CRS, str]) -> bool:
    """Evaluates whether two coordinate reference systems are the same.

    Args:
        crs1: the first CRS, from a rasterio dataset, a GeoDataFrame, or a string with projection parameters.
        crs2: the second CRS, from the same sources above.

    Returns:
        matches: Boolean for whether the CRS match.
    """
    # normalize string inputs via rasterio
    if type(crs1) is str:
        crs1 = string_to_crs(crs1)
    if type(crs2) is str:
        crs2 = string_to_crs(crs2)

    matches = crs1 == crs2

    return matches

def stack_geodataframes(presence: Vector, background: Vector, add_class_label: bool = False) -> gpd.GeoDataFrame:
    """Concatenate geometries from two GeoSeries/GeoDataFrames.

    Args:
        presence: presence geometry (y=1) locations
        background: background geometry (y=0) locations
        add_class_label: add a column labeling the y value for each point

    Returns:
        merged GeoDataFrame with all geometries projected to the same crs.
    """
    presence = presence.copy()
    background = background.copy()
    
    validate_gpd(presence)
    validate_gpd(background)

    # cast to geodataframes
    if isinstance(presence, gpd.GeoSeries):
        presence = presence.to_frame("geometry")
    if isinstance(background, gpd.GeoSeries):
        background = background.to_frame("geometry")

    # handle projection mismatch
    crs = presence.crs
    if crs_match(presence.crs, background.crs):
        # explicitly set the two to exactly matching crs as geopandas
        # throws errors if there's any mismatch at all
        background.crs = presence.crs
    else:
        background.to_crs(crs, inplace=True)

    presence['class'] = 1
    background['class'] = 0

    matching = [col for col in presence.columns if col in background.columns]
    assert len(matching) > 0, "no matching columns found between data frames"

    # Reset index to avoid duplicate indices
    presence = presence[matching].reset_index(drop=True)
    background = background[matching].reset_index(drop=True)

    merged = pd.concat((presence, background), axis=0, ignore_index=True)
    gdf = gpd.GeoDataFrame(merged, crs=presence.crs)

    return gdf

def convert_geometry_collection_to_multipolygon(gdf, buffer_distance=0.5):
    """Convert GeometryCollection to MultiPolygon in the entire GeoDataFrame, buffering points and lines if necessary."""

    def process_geometry(geometry):
        if isinstance(geometry, GeometryCollection):
            polygons = [geom.buffer(buffer_distance) if isinstance(geom, (Point, LineString, MultiPoint, MultiLineString)) 
                        else geom 
                        for geom in geometry.geoms if isinstance(geom, (Polygon, MultiPolygon, Point, LineString, MultiPoint, MultiLineString))]

            if len(polygons) == 1:
                return MultiPolygon(polygons) if isinstance(polygons[0], Polygon) else polygons[0]
            elif len(polygons) > 1:
                return MultiPolygon(polygons)
            else:
                return None
        return geometry

    gdf['geometry'] = gdf['geometry'].apply(process_geometry)
    
    return gdf

def checkerboard_split(
    points: Vector, grid_size: float, buffer: float = 0, bounds: Tuple[float, float, float, float] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create train/test splits with a spatially-gridded checkerboard.

    Args:
        points: point-format GeoSeries or GeoDataFrame
        grid_size: the height and width of each checkerboard side to split
            data using. Should match the units of the points CRS
            (i.e. grid_size=1000 is a 1km grid for UTM data)
        buffer: add an x/y buffer around the initial checkerboard bounds
        bounds: instead of deriving the checkerboard bounds from `points`,
            use this tuple of [xmin, ymin, xmax, ymax] values.

    Returns:
        (train_points, test_points) split using a checkerboard grid.
    """
    if isinstance(points, gpd.GeoSeries):
        points = points.to_frame("geometry")

    bounds = points.total_bounds if bounds is None else bounds
    xmin, ymin, xmax, ymax = bounds

    x0s = np.arange(xmin - buffer, xmax + buffer + grid_size, grid_size)
    y0s = np.arange(ymin - buffer, ymax + buffer + grid_size, grid_size)

    train_cells = []
    test_cells = []
    for idy, y0 in enumerate(y0s):
        offset = 0 if idy % 2 == 0 else 1
        for idx, x0 in enumerate(x0s):
            cell = box(x0, y0, x0 + grid_size, y0 + grid_size)
            cell_type = 0 if (idx + offset) % 2 == 0 else 1
            if cell_type == 0:
                train_cells.append(cell)
            else:
                test_cells.append(cell)

    grid_crs = points.crs
    train_grid = gpd.GeoDataFrame(geometry=train_cells, crs=grid_crs)
    test_grid = gpd.GeoDataFrame(geometry=test_cells, crs=grid_crs)
    train_points = (
        gpd.sjoin(points, train_grid, how="left", predicate="within")
        .dropna()
        .drop(columns="index_right")
        .reset_index(drop=True)
    )
    test_points = (
        gpd.sjoin(points, test_grid, how="left", predicate="within")
        .dropna()
        .drop(columns="index_right")
        .reset_index(drop=True)
    )

    return train_points, test_points

def nearest_point_distance(
    points1: Vector, points2: Vector = None, n_neighbors: int = 1, cpu_count: int = -1
) -> np.ndarray:
    """Compute the average euclidean distance to the nearest point in a series.

    Args:
        points1: return the closest distance *from* these points
        points2: return the closest distance *to* these points
            if None, compute the distance to the nearest points
            in the points1 series
        n_neighbors: compute the average distance to the nearest n_neighbors.
            set to -1 to compute the distance to all neighbors.
        cpu_count: number of cpus to use for estimation.
            -1 uses all cores

    Returns:
        array of shape (len(points),) with the distance to
            each point's nearest neighbor
    """
    if points1.crs.is_geographic:
        warnings.warn("Computing distances using geographic coordinates is bad")

    pta1 = np.array(list(zip(points1.geometry.x, points1.geometry.y)))
    k_offset = 1

    if points2 is None:
        pta2 = pta1
        k_offset += 1

    else:
        pta2 = np.array(list(zip(points2.geometry.x, points2.geometry.y)))
        if not crs_match(points1.crs, points2.crs):
            warnings.warn("CRS mismatch between points")

    if n_neighbors < 1:
        n_neighbors = len(pta2) - k_offset

    tree = KDTree(pta1)
    k = np.arange(n_neighbors) + k_offset
    distance, idx = tree.query(pta2, k=k, workers=cpu_count)

    return distance.mean(axis=1)

def distance_weights(points: Vector, n_neighbors: int = -1, center: str = "median", cpu_count: int = -1) -> np.ndarray:
    """Compute sample weights based on the distance between points.

    Assigns higher scores to isolated points, lower scores to clustered points.

    Args:
        points: point-format GeoSeries or GeoDataFrame
        n_neighbors: compute weights based on average distance to the nearest n_neighbors
            set to -1 to compute the distance to all neighbors.
        center: rescale the weights to center the mean or median of the array on 1
            accepts either 'mean' or 'median' as input.
            pass None to ignore.
        cpu_count: number of cpus to use for estimation.
            -1 uses all cores

    Returns:
        array of shape (len(points),) with scaled sample weights. Scaling
            is performed by dividing by the maximum value, preserving the
            relative scale of differences between the min and max distance.
    """
    distances = nearest_point_distance(points, n_neighbors=n_neighbors, cpu_count=cpu_count)
    weights = distances / distances.max()

    if center is not None:
        if center.lower() == "mean":
            weights /= weights.mean()

        elif center.lower() == "median":
            weights /= np.median(weights)

    return weights

def get_tss(y_test, predictions):
    """
    Calculate the True Skill Statistic (TSS) using prediction results and y_test.
    
    Parameters:
    y_test (array-like): True labels.
    predictions (array-like): Predicted labels.
    
    Returns:
    float: TSS value.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    tss = sensitivity + specificity - 1
    return tss

def visualise_feature_importances(model, feature_columns):
    """
    Visualize the feature importances from a trained RandomForest model and return the top 5 feature indexes.
    
    Parameters:
    model (RandomForestClassifier): Trained RandomForest model.
    feature_columns (list): List of feature column names.
    
    Returns:
    list: Indexes of the top 5 important features.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top5_indices = indices[:5]

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    return top5_indices

def plot_results(gdf, bird_name, save=True):
    """
    Plot and save anomaly detection results.
    """
    fig, axes = plt.subplots(1, 4, figsize=(19, 6))
    columns = ['rf_probability', 'gb_probability', 'me_probability', 'probability']
    titles = [
        'Random Forest', 'Histogram Gradient Boosting', 
        'MaxEnt', 'Combined Average'
    ]
    
    for ax, col, title in zip(axes, columns, titles):
        gdf.plot(column=col, cmap='coolwarm', legend=True, markersize=5, ax=ax)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

    if save==True:
        os.makedirs('data_results/images2', exist_ok=True)
        fig.savefig(f'data_results/images2/{bird_name}.png')