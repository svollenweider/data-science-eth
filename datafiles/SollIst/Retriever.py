from bs4 import BeautifulSoup
import urllib.request as url
import re
import os
 
year='2017'
website = "https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd_"+year

html_page = url.urlopen(website)
soup = BeautifulSoup(html_page)
links = []
 
for link in soup.findAll('a', attrs={'href': re.compile("^https://")}):
    name = link.get('href')
    if 'fahrzeitensollist' in name and name.endswith('.csv'):
        links.append(link.get('href'))

i = 0
print(str(round(i/len(links),3)*100)+' %')
for link in links:
    url.urlretrieve(link, 'SollIst/Results/'+os.path.split(link)[1])
    i += 1
    print(str(round(i/len(links),3)*100)+' %')