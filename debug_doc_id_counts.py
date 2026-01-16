import chromadb
from pathlib import Path

PERSIST_DIR = r"C:\Users\madis\Projects\rag-lab\gitversions\RAG-pipeline-v2trial\stage_3_chroma_test5"
COLLECTION = "v1_chunks"
DOC_ID = "a5bee0ef92a3dd632aa678d6"

client = chromadb.PersistentClient(path=str(Path(PERSIST_DIR)))
coll = client.get_collection(name=COLLECTION)

try:
    res = coll.get(where={"doc_id": DOC_ID}, include=[])
except TypeError:
    res = coll.get(where={"doc_id": DOC_ID})
ids = res.get("ids", [])
print("doc_id:", DOC_ID)
print("chunk_count:", len(ids))
print("sample_ids:", ids[:10])
