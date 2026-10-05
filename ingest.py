from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import re
import arxiv
import chromadb
import voyageai

vo = voyageai.Client()  # reads VOYAGE_API_KEY
db = chromadb.PersistentClient(path="./astro_db")
col = db.get_or_create_collection("astro", metadata={"hnsw:space": "cosine"})

def base_id(result):
    # strip version suffix so v1/v2 of a paper don't duplicate
    return re.sub(r"v\d+$", "", result.get_short_id())

def flush(batch):
    texts = [f"{r.title}. {r.summary}" for r in batch]
    emb = vo.embed(texts, model="voyage-3.5", input_type="document").embeddings
    col.upsert(  # upsert makes re-running safe
        ids=[base_id(r) for r in batch],
        embeddings=emb,
        documents=texts,
        metadatas=[{
            "title": r.title,
            "url": r.entry_id,
            "published": r.published.strftime("%Y-%m-%d"),
            "category": r.primary_category,
            "authors": ", ".join(a.name for a in r.authors[:5]),
        } for r in batch],
    )

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
search = arxiv.Search(
    query="cat:astro-ph.* AND submittedDate:[202601010000 TO 202610052359]",
    max_results=2000,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

batch = []
for r in client.results(search):
    batch.append(r)
    if len(batch) == 100:   # stays under Voyage's per-request limit
        flush(batch)
        batch = []
        print("indexed 100 more")
if batch:
    flush(batch)