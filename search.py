from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import sys
import anthropic
import chromadb
import voyageai

vo = voyageai.Client()
llm = anthropic.Anthropic()
col = chromadb.PersistentClient(path="./astro_db").get_collection("astro")

def retrieve(query, k=8, category=None, since=None):

    where = {}
    if category:
        where["category"] = category
    
    if since:
        where["published"] = {"$gte: since"}

    q = vo.embed([query], model="voyage-3.5", input_type="query").embeddings
    return col.query(
        query_embeddings=q,
        n_results=k,
        where=where or None,
    )

#def answer(query, **filters):
    #res = retrieve(query, **filters)
    #docs, metas = res["documents"][0], res["metadatas"][0]
    #context = "\n\n".join(
        #f"[{i+1}] {m['title']} ({m['published']}, {m['category']})\n{d}"
        #for i, (d,m) in enumerate(zip(docs, metas))
    #)
    #msg = llm.messages.create(
        #model= "claude-sonnet-5-5",
        #max_tokens=1000,
        #messages=[{"role": "user", "content": (
            #"Answer using only the arXiv abstracts below. Cite with [n]. "
            #"If they don't cover the question, say so. \n\n"
            #f"{context}\n\nQuestion: {query}"
        #)}],
    #)

    #sources = "\n".join(f"[{i+1}] {m['title']} - {m['url']}" for i, m in enumerate(metas))
    #return msg.content[0].text + "\n\nSources:\n" + sources

def show_results(query, **filters):
    res = retrieve(query, **filters)
    for i, (d, m, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
        print(f"[{i+1}] {m['title']}")
        print(f"    {m['published']} | {m['category']} | similarity: {1 - dist:.3f}")
        print(f"    {m['url']}")
        print(f"    {d[:300]}...\n")

def answer(query, **filters):
    res = retrieve(query, **filters)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(
        f"[{i+1}] {m['title']} ({m['published']}, {m['category']})\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas))
    )
    msg = llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": (
            "Answer using only the arXiv abstracts below. Cite with [n]. "
            "If they don't cover the question, say so.\n\n"
            f"{context}\n\nQuestion: {query}"
        )}],
    )
    sources = "\n".join(f"[{i+1}] {m['title']} - {m['url']}" for i, m in enumerate(metas))
    return msg.content[0].text + "\n\nSources:\n" + sources

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--no-llm":
        show_results(" ".join(args[1:]))
    else:
        print(answer(" ".join(args)))#

#if __name__ == "__main__":
    #print(answer(" ".join(sys.argv[1:])))
