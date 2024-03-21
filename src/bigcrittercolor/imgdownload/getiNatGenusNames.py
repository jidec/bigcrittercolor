import requests

def getiNatGenusNames(parent_taxon):
    # Initial request to get the parent taxon ID
    response = requests.get('https://api.inaturalist.org/v1/taxa', params={'q': parent_taxon})
    data = response.json()
    taxon_id = None
    for result in data.get('results', []):
        if result['name'].lower() == parent_taxon.lower():
            taxon_id = result['id']
            break  # Exit the loop once the parent taxon is found

    if taxon_id is None:
        return []  # Return an empty list if the parent taxon was not found

    genera = []
    page = 1
    while True:
        requests_str = f'https://api.inaturalist.org/v1/taxa?taxon_id={taxon_id}&is_active=true&rank_level=20&per_page=200&page={page}'
        response = requests.get(requests_str)
        data = response.json()

        results = data.get('results', [])
        if not results:
            break  # Exit the loop if there are no more results

        for result in results:
            genera.append(result['name'])

        page += 1  # Increment the page number for the next iteration

    print(f'Total genera found: {len(genera)}')
    return genera