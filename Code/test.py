import requests
from bs4 import BeautifulSoup

r = requests.get("https://cit.ac.in")
soup = BeautifulSoup(r.content, 'html5lib')
print(soup)