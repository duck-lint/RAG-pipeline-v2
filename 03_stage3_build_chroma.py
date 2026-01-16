from __future__ import annotations

import argparse
import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
import torch

PIPELINE_VERSION = "v1"
STAGE3_VERSION = "v0.1"

CHROMA_META_KEYS = [
    "doc_id",
    "chunk_id",
    "chunk_key",
    "chunk_hash",
    "chunk_anchor",
    "chunk_title",
    "heading_path_str",
    "chunk_index",
    "rel_path",
    "source_uri",
    "cleaned_text",
    "entry_date",
    "source_date",
    "source_hash",
    "content_hash",
    "embed_model",
    "embed_dim",
    "chunker_version",
    "doc_type",
    "sensitivity",
    "folder",
]

def stable_settings_hash(d: Dict[str, Any]) -> str:
    # Stable hash of settings (sorted keys) so runs are comparable
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def batch(iterable: List[Any], n: int) -> List[List[Any]]:
    return [iterable[i : i + n] for i in range(0, len(iterable), n)]

def find_existing_ids(collection: Any, ids: List[str], batch_size: int) -> List[str]:
    existing: List[str] = []
    for idxs in batch(list(range(len(ids))), batch_size):
        batch_ids = [ids[i] for i in idxs]
        try:
            res = collection.get(ids=batch_ids, include=[])
        except TypeError:
            res = collection.get(ids=batch_ids)
        existing.extend(res.get("ids", []))
    return existing

def get_existing_meta_by_id(collection: Any, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    try:
        res = collection.get(ids=ids, include=["metadatas"])
    except TypeError:
        res = collection.get(ids=ids)
    got_ids = res.get("ids", [])
    metas = res.get("metadatas", []) or []
    return {i: m for i, m in zip(got_ids, metas) if m is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", type=str, default="stage_2_chunks.jsonl", help="default=stage_2_chunks.jsonl")
    ap.add_argument("--persist_dir", type=str, default="stage_3_chroma", help="default=stage_3_chroma")
    ap.add_argument("--db_dir", type=str, help="(deprecated) use --persist_dir instead")
    ap.add_argument("--collection", type=str, default="v1_chunks", help="default=v1_chunks")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="default=sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda | default=auto")
    ap.add_argument("--batch_size", type=int, default=32, help="default=32")
    ap.add_argument("--reset_db", action="store_true", help="Delete persist_dir before building (rebuild only)")
    ap.add_argument("--mode", type=str, choices=["rebuild", "append", "upsert"], default="upsert")
    ap.add_argument("--skip_unchanged", action="store_true", help="When upserting, skip chunks whose hash hasn't changed")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print(f"[stage_03_chroma] args: {args}")

    chunks_path = Path(args.chunks_jsonl).resolve()
    persist_dir = Path(args.persist_dir).resolve()
    if args.db_dir:
        persist_dir = Path(args.db_dir).resolve()

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
    if not args.collection or not args.collection.strip():
        raise ValueError("--collection must be a non-empty name")

    # Decide device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    settings = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "chunks_jsonl": str(chunks_path),
        "persist_dir": str(persist_dir),
        "collection": args.collection,
        "embed_model": args.embed_model,
        "device": device,
        "batch_size": args.batch_size,
        "reset_db": args.reset_db,
        "mode": args.mode,
    }
    settings_hash = stable_settings_hash(settings)

    rows = load_jsonl(chunks_path)
    ids = [r["metadata"]["chunk_id"] for r in rows]
    docs = [r["text"] for r in rows]
    metas = []
    for r in rows:
        m = r["metadata"]
        heading_path = m.get("heading_path") or []
        if isinstance(heading_path, list):
            heading_path_str = " > ".join(heading_path)
        else:
            heading_path_str = str(heading_path)
        base = {k: m.get(k) for k in CHROMA_META_KEYS if k != "heading_path_str" and m.get(k) is not None}
        base["heading_path_str"] = heading_path_str
        metas.append(base)

    missing_meta = []
    for idx, r in enumerate(rows):
        m = r.get("metadata", {})
        required_keys = ["chunk_id", "chunk_key", "chunk_hash", "source_uri", "chunk_index", "cleaned_text"]
        missing = False
        for k in required_keys:
            if k not in m:
                missing = True
                break
            if m.get(k) in (None, ""):
                missing = True
                break
        heading_path = m.get("heading_path") or []
        if isinstance(heading_path, list):
            heading_path_str = " > ".join(heading_path)
        else:
            heading_path_str = str(heading_path)
        if heading_path_str is None:
            missing = True
        if missing:
            missing_meta.append(idx)
            if len(missing_meta) >= 5:
                break
    if missing_meta:
        raise ValueError(f"Missing required chunk metadata in rows: {missing_meta}")

    # Basic sanity
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id detected in stage_2_chunks.jsonl")

    print(
        f"[stage_3] start persist_dir={persist_dir} | collection={args.collection} | "
        f"mode={args.mode} | docs={len(docs)} | chunks={len(rows)}"
    )
    print("[stage_3] ---- input summary ----")
    print(f"[stage_3] chunks_jsonl={chunks_path}")
    print(f"[stage_3] rows={len(rows)}")
    print(f"[stage_3] embed_model={args.embed_model}")
    print(f"[stage_3] device={device} | cuda_available={torch.cuda.is_available()}")
    print(f"[stage_3] settings_hash={settings_hash}")

    if args.dry_run:
        print("[stage_3] dry_run=True (not embedding / not writing DB)")
        return

    # Reset persist directory if requested (rebuild mode only)
    if args.reset_db and args.mode == "rebuild" and persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"[stage_3] deleted persist_dir: {persist_dir}")

    persist_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = SentenceTransformer(args.embed_model, device=device)

    # Create Chroma persistent client
    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.mode == "rebuild":
        try:
            client.delete_collection(name=args.collection)
            print(f"[stage_3] deleted existing collection: {args.collection}")
        except Exception:
            pass
        collection = client.create_collection(
            name=args.collection,
            metadata={
                "pipeline_version": PIPELINE_VERSION,
                "stage3_version": STAGE3_VERSION,
                "embed_model": args.embed_model,
                "device": device,
                "settings_hash": settings_hash,
            },
        )
    else:
        try:
            collection = client.get_collection(name=args.collection)
        except Exception:
            collection = client.create_collection(
                name=args.collection,
                metadata={
                    "pipeline_version": PIPELINE_VERSION,
                    "stage3_version": STAGE3_VERSION,
                    "embed_model": args.embed_model,
                    "device": device,
                    "settings_hash": settings_hash,
                },
            )
            print(f"[stage_3] created collection: {args.collection}")
        else:
            print(f"[stage_3] using existing collection: {args.collection}")
            coll_meta = getattr(collection, "metadata", None) or {}
            coll_model = coll_meta.get("embed_model")
            coll_hash = coll_meta.get("settings_hash")
            if coll_model and coll_model != args.embed_model:
                raise ValueError(
                    "Collection embed_model mismatch. "
                    "Use --mode rebuild or a new --collection name."
                )
            if coll_hash and coll_hash != settings_hash:
                raise ValueError(
                    "Collection settings_hash mismatch. "
                    "Use --mode rebuild or a new --collection name."
                )

    if args.mode == "append":
        existing = find_existing_ids(collection, ids, batch_size=min(args.batch_size, 256))
        if existing:
            sample = existing[:10]
            raise ValueError(
                f"Append mode found {len(existing)} duplicate ids. Sample: {sample}"
            )

    # Embed and ingest in batches
    total = 0
    for idxs in batch(list(range(len(docs))), args.batch_size):
        batch_docs = [docs[i] for i in idxs]
        batch_ids = [ids[i] for i in idxs]
        batch_metas = [metas[i] for i in idxs]

        if args.mode == "upsert":
            if args.skip_unchanged:
                existing_meta = get_existing_meta_by_id(collection, batch_ids)
                keep_ids: List[str] = []
                keep_docs: List[str] = []
                keep_metas: List[Dict[str, Any]] = []
                for i, cid in enumerate(batch_ids):
                    prev = existing_meta.get(cid)
                    if prev and prev.get("chunk_hash") == batch_metas[i].get("chunk_hash"):
                        continue
                    keep_ids.append(cid)
                    keep_docs.append(batch_docs[i])
                    keep_metas.append(batch_metas[i])
                if not keep_ids:
                    continue
                batch_ids = keep_ids
                batch_docs = keep_docs
                batch_metas = keep_metas
            embeddings = model.encode(
                batch_docs,
                batch_size=min(args.batch_size, len(batch_docs)),
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine-friendly
                show_progress_bar=False,
            )
            embeddings_list = embeddings.tolist()
            if hasattr(collection, "upsert"):
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings_list,
                )
            else:
                existing = find_existing_ids(collection, batch_ids, batch_size=len(batch_ids))
                if existing:
                    collection.delete(ids=existing)
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings_list,
                )
        else:
            embeddings = model.encode(
                batch_docs,
                batch_size=min(args.batch_size, len(batch_docs)),
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine-friendly
                show_progress_bar=False,
            )
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=embeddings.tolist(),
            )
        total += len(batch_docs)

    print("[stage_3] ---- build summary ----")
    print(f"[stage_3] wrote collection={args.collection} | total_added={total}")
    print(f"[stage_3] persist_dir={persist_dir}")

    # Write run manifest
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "stage3_version": STAGE3_VERSION,
        "settings": settings,
        "settings_hash": settings_hash,
        "counts": {"chunks": len(rows), "added": total},
    }
    manifest_path = Path("run_manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage_3] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
