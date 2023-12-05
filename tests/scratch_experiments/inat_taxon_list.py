import requests

def get_species_list(taxon_id):
    url = f"https://api.inaturalist.org/v1/taxa/{taxon_id}/children"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        species = [taxon['name'] for taxon in data['results']]
        return species
    else:
        return f"Error: Unable to fetch data. Status code: {response.status_code}, Response: {response.text}"

# Example usage
taxon_id = 47792  # replace with your Taxon ID
species_list = get_species_list(taxon_id)
print(species_list)