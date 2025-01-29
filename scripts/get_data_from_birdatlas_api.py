import geopandas as gpd
import requests
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from retry import retry

input_gpkg = "YKJ10km_polygons\YKJ_Corine_birds2.gpkg"
output_gpkg = "YKJ10km_polygons\YKJ_Corine_birds3.gpkg"

print("Reading data...")
gdf = gpd.read_file(input_gpkg)
coordinates = gdf['coordinates'].unique()
gdf['activity_category'] = None

@retry(tries=5, delay=2)
def fetch_activity_category(coordinate):
    url = f"https://atlas-api.2.rahtiapp.fi/api/v1/grid/{coordinate}/atlas"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    data = response.json()
    return coordinate, data['activityCategory']['value']

print("Fetching data from API...")
results = []

with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_coordinate = {executor.submit(fetch_activity_category, coord): coord for coord in coordinates}
    for future in tqdm.tqdm(as_completed(future_to_coordinate), total=len(coordinates)):
        try:
            coordinate, activity_category = future.result()
            results.append((coordinate, activity_category))
        except Exception as e:
            print(f"Error fetching data for coordinate {future_to_coordinate[future]}: {e}")

# Store activity categories in the GeoDataFrame
print("Storing data...")
for coordinate, activity_category in results:
    gdf.loc[gdf['coordinates'] == coordinate, 'activity_category'] = activity_category

gdf.to_file(output_gpkg, driver='GPKG')




