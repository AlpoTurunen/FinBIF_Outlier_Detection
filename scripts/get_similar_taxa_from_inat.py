from pyinaturalist import *
from load_data import fetch_json_with_retry

def get_similar_species_id_from_inat(finbif_taxon_id, access_token, number_of_species=5):
    """
    Fetches the closest similar species ID from the FinBIF database by leveraging iNaturalist's 
    similar species API. 
    
    Parameters:
        finbif_taxon_id (str): The FinBIF taxon ID of the species.
        access_token (str): Access token for the FinBIF API.
        number_of_species (int): Number to describe how many similar species IDs you want. Default 5
        
    Returns:
        str: The FinBIF taxon ID of the most similar species.
    """

    # Get scientific name for the given FinBIF taxon ID
    url = f'https://api.laji.fi/v0/taxa/{finbif_taxon_id}?lang=en&langFallback=true&maxLevel=0&selectedFields=scientificName&access_token={access_token}'
    results = fetch_json_with_retry(url)
    scientific_name = results['scientificName']

    # Search for the species in iNaturalist using the scientific name
    response = get_taxa(q=scientific_name)['results']
    inat_id = response[0]['id']

    # Retrieve similar species from iNaturalist's Finnish observations based on the matched species
    similar_species_url = f'https://api.inaturalist.org/v1/identifications/similar_species?place_id=7020&taxon_id={inat_id}'
    similar_species_json = fetch_json_with_retry(similar_species_url)
    
    if not similar_species_json: # Try to search similar species globally
        similar_species_url = f'https://api.inaturalist.org/v1/identifications/similar_species?&taxon_id={inat_id}'
        similar_species_json = fetch_json_with_retry(similar_species_url)
        
    most_similar_taxon_names = []
    for i in range(number_of_species):
        try:
            taxon_name = similar_species_json['results'][i]['taxon']['name'] # TODO: Add percentages maybe?
            most_similar_taxon_names.append(taxon_name)
        except:
            print(f"Only {i} similar taxons found instead of {number_of_species}")
            break

    print(f"Most similar taxon names on iNaturalist are: {most_similar_taxon_names}")

    # Find the FinBIF ID for the similar species name identified on iNaturalist
    most_similar_taxon_finbif_ids = []
    for taxon_name in most_similar_taxon_names:
        finbif_search_url = f"https://api.laji.fi/v0/taxa/search?query={taxon_name}&limit=10&onlySpecies=false&onlyFinnish=false&onlyInvasive=false&observationMode=false&access_token={access_token}"
        results = fetch_json_with_retry(finbif_search_url)
        taxon_id = results[0]['id']
        most_similar_taxon_finbif_ids.append(taxon_id)

    return most_similar_taxon_finbif_ids
