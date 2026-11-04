from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
load_dotenv(Path(__file__).parent / ".env")

import re
import arxiv
import chromadb
import voyageai

vo = voyageai.Client()
db = chromadb.PersistentClient(path="./astro_db")
col = db.get_or_create_collection("astro", metadata={"hnsw:space": "cosine"})

MAX_PER_WINDOW = 5 # was 2000

def base_id(result):

    return re.sub(r"v\d+$", "", result.get_short_id())

def flush(batch):
    texts = [f"{r.title}. {r.summary}" for r in batch]
    emb = vo.embed(texts, model="voyage-3.5", input_type="document").embeddings
    col.upsert(
        ids=[base_id(r) for r in batch],
        embeddings=emb,
        documents=texts,
        metadatas=[{
            "title": r.title,
            "url": r.entry_id,
            "published": r.published.strftime("%Y-%m-%d"),
            "published_int": int(r.published.strftime("%Y%m%d")),
            "category": r.primary_category,
            "authors": ", ".join(a.name for a in r.authors[:5]),
        } for r in batch],
    )

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)

def ingest_window(start, end):

    search = arxiv.Search(
        query=f"cat:astro-ph.* AND submittedDate:[{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}]",
        max_results=MAX_PER_WINDOW,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    batch = []
    paper_counter = 0
    for r in client.results(search):
        batch.append(r)
        paper_counter += 1

        if len(batch) == 100:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    
    return paper_counter

cursor = datetime(2026, 1, 1)
while cursor < datetime(2026, 1, 22):
    window_end = cursor + timedelta(days=7) - timedelta(minutes=1)
    n_papers = ingest_window(cursor, window_end)
    print(f"{cursor} -> {window_end}: {n_papers}")
    if n_papers == MAX_PER_WINDOW:
        print("Warning : More papers available than current window")

    cursor = cursor + timedelta(days = 7)


