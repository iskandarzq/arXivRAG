Open a python shell :
venv\Scripts\python

And then :
from search import retrieve

r = retrieve("dark matter", category="astro-ph.GA", since="2026-09-20")
len(r["ids"][0])

To search (doesnt work with filters for now) : python search.py --no-llm "dark matter", since="2026-09-20"
