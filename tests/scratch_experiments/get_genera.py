import requests

def get_taxon_id_for_odonata():
    """Get the taxon ID for Odonata."""
    response = requests.get('https://api.inaturalist.org/v1/taxa', params={'q': 'Odonata'})
    data = response.json()
    for result in data.get('results', []):
        if result['name'].lower() == 'odonata':
            return result['id']
    return None

def get_odonata_genera(taxon_id):
    """Get a list of genus names within the Odonata order."""
    genera = []

    requests_str = 'https://api.inaturalist.org/v1/taxa?taxon_id=' + str(taxon_id) + "&is_active=true&rank_level=20&per_page=1000"
    print(requests_str)
    response = requests.get(requests_str)
    data = response.json()

    results = data.get('results', [])
    for result in results:
        genera.append(result['name'])

    return genera

taxon_id = get_taxon_id_for_odonata()
genera = get_odonata_genera(taxon_id)
print(genera)