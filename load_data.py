import geopandas as gpd
import pandas as pd
import requests
import time
import urllib

def get_last_page(url):
    """
    Get the last page number from the API response with retry logic.

    Parameters:
    url (str): The URL of the Warehouse API endpoint.

    Returns:
    int: The last page number. Returns None if all retries fail.
    """
    attempt = 0
    max_retries = 3
    delay = 10
    while attempt < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            api_response = response.json()
            return api_response.get("lastPage", None)
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error retrieving last page from {url}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
    print(f"Failed to retrieve last page from {url} after {max_retries} attempts.")
    return None

def download_page(url, page_no):
    """
    Download data from a specific page of the API with retry logic. This is in separate function to speed up multiprocessing.

    Parameters:
    url (str): The URL of the Warehouse API endpoint.
    page_no (int): The page number to download.

    Returns:
    geopandas.GeoDataFrame: The downloaded data as a GeoDataFrame.
    """
    # Load data
    attempt = 0
    max_retries = 5
    delay = 30
    url = url.replace('page=1', f'page={page_no}')
    while attempt < max_retries:
        try:
            gdf = gpd.read_file(url)   
            return gdf 
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason} for {url}. Retrying in {delay} seconds...")
        except Exception as e:
            print(f"Error downloading page {page_no}: {e}. Retrying in {delay} seconds...")
        time.sleep(delay)
        attempt += 1

    # Return an empty GeoDataFrame in case of too many errors
    print(f"Failed to download data from page {page_no} after {max_retries} attempts.")
    return gpd.GeoDataFrame()

def get_occurrence_data(data_url, pages=10):
    """
    Retrieve occurrence data from the API.

    Parameters:
    data_url (str): The URL of the API endpoint.
    pages (str or int, optional): Number of pages to retrieve. Defaults to "all".

    Returns:
    geopandas.GeoDataFrame: The retrieved occurrence data as a GeoDataFrame.
    """
    
    if pages == 'all':
        endpage = get_last_page(data_url)
        print(f"Loading {endpage} pages..")
    else:
        endpage = int(pages)
    
    gdf = gpd.GeoDataFrame()
    startpage = 1

    for page_no in range(startpage,endpage+1):
        next_gdf = download_page(data_url, page_no)
        gdf = pd.concat([gdf, next_gdf], ignore_index=True)
        print(f"Page {page_no} downloaded")

    return gdf

def fetch_json_with_retry(url, max_retries=5, delay=30):
    """
    Fetches JSON data from an API URL with retry logic.
    
    Parameters:
    url (str): The API URL to fetch JSON data from.
    max_retries (int): The maximum number of retry attempts in case of failure.
    delay (int): The delay between retries in seconds.
    
    Returns:
    dict: Parsed JSON data from the API as a dictionary, or None if the request fails.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
    print(f"Failed to retrieve data from {url} after {max_retries} attempts.")
    return None