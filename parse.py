import requests
from bs4 import BeautifulSoup
import sys

URL = sys.args[1]

page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

fulltext = ""

for p in soup.find_all("p"):
    fulltext.appned(p.get_text())

print(fulltext)