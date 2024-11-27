import requests

# URL of the webpage you want to download
url = 'https://www.inaturalist.org/observations/5556679'

# Send a request to the webpage
response = requests.get(url)

# Extract the HTML text
html_text = response.text

# Optionally, save the HTML to a file
with open('webpage.html', 'w', encoding='utf-8') as file:
    file.write(html_text)

# Display the HTML text
print(html_text)