from pathlib import Path
import argparse
import re
from common import (
    read_text,
    strip_yaml_frontmatter,
    parse_yaml_frontmatter,
    parse_source_date,
    normalize_markdown_light,
    replace_wikilinks_and_collect,
    sha256_bytes,
    sha256_text,
    write_jsonl,
    split_into_sections,
    parse_date_field,
)

CHUNKER_VERSION = "v0.1"

def strip_leading_heading_lines(p: str) -> str:
    lines = p.splitlines()
    out = []
    skipping = True
    for ln in lines:
        if skipping and re.match(r"^\s{0,3}#{1,6}\s+", ln):
            continue
        skipping = False
        out.append(ln)
    return "\n".join(out).strip()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0_raw", type=str, required=True, help="Path to stage_0_raw/*.md")
    ap.add_argument("--out_jsonl", type=str, default="stage_2_chunks.jsonl", help="default=stage_2_chunks.jsonl")
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="default=sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed_dim", type=int, default=384, help="default=384")
    ap.add_argument("--max_chars", type=int, default=2500, help="Split section if longer than this (V1 safety)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stage1_dir", type=str, default="stage_1_clean")
    ap.add_argument("--prefer_stage1", action="store_true", help="Chunk from stage_1_clean if available")
    args = ap.parse_args()

    print(f"[stage_02_chunk] args: {args}")

    src = Path(args.stage0_raw).resolve()
    raw_bytes = src.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    print(f"[stage_02_chunk] src={src}")

    body, yaml_block = strip_yaml_frontmatter(raw_text)
    meta = parse_yaml_frontmatter(yaml_block or "")
    stage1_path = Path(args.stage1_dir).resolve() / f"{src.stem}.clean.txt"

    if args.prefer_stage1:
        if not stage1_path.exists():
            raise FileNotFoundError(f"--prefer_stage1 set but stage1 file missing: {stage1_path}")
        chunk_input = read_text(stage1_path)
    else:
        # fallback: chunk from stage0 body (still works)
        chunk_input = normalize_markdown_light(body_md)

    doc_id = str(meta.get("uuid") or "").strip() or sha256_bytes(raw_bytes)[:24]  # V1 fallback
    rel_path = str(Path("stage_0_raw") / src.name)  # simple V1; refine later
    entry_date = parse_date_field(meta, "journal_entry_date")
    source_date = parse_date_field(meta, "note_creation_date")
    source_hash = sha256_bytes(raw_bytes)

    sections = split_into_sections(chunk_input)

    rows = []
    total_chunks = 0

    for anchor, section_title, section_raw in sections:
        # normalize then replace links per chunk
        normalized = normalize_markdown_light(section_raw)

        # Split into paragraphs (blank-line separated)
        paragraphs_raw = [p.strip() for p in normalized.split("\n\n") if p.strip()]

        # Drop empty sections (this also drops "header-only" sections because header lines are no longer in section_raw)
        if not paragraphs_raw:
            continue

        parts: list[str] = []
        parts_links: list[list[dict]] = []

        for para_raw in paragraphs_raw:
            if len(para_raw) <= args.max_chars:
                chunk_text, chunk_links = replace_wikilinks_and_collect(para_raw)
                parts.append(chunk_text.strip() + "\n")
                parts_links.append(chunk_links)
            else:
                # deterministic split for huge paragraphs
                buf: list[str] = []
                cur = 0
                for piece in re.split(r"(?<=[.!?])\s+", para_raw):
                    if cur + len(piece) + 1 > args.max_chars and buf:
                        combined_raw = " ".join(buf)
                        chunk_text, chunk_links = replace_wikilinks_and_collect(combined_raw)
                        parts.append(chunk_text.strip() + "\n")
                        parts_links.append(chunk_links)
                        buf = []
                        cur = 0
                    buf.append(piece)
                    cur += len(piece) + 1

                if buf:
                    combined_raw = " ".join(buf)
                    chunk_text, chunk_links = replace_wikilinks_and_collect(combined_raw)
                    parts.append(chunk_text.strip() + "\n")
                    parts_links.append(chunk_links)

        for idx, chunk_text in enumerate(parts):
            chunk_id = f"{doc_id}::{anchor}::{idx}"
            content_hash = sha256_text(chunk_text)

            rows.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_anchor": anchor,
                    "chunk_title": section_title,
                    "chunk_index": idx,
                    "rel_path": rel_path,
                    "entry_date": entry_date,
                    "source_date": source_date,
                    "source_date": source_date,
                    "source_hash": source_hash,
                    "content_hash": content_hash,
                    "embed_model": args.embed_model,
                    "embed_dim": args.embed_dim,
                    "chunker_version": CHUNKER_VERSION,
                    "out_links": parts_links[idx],
                }
            })
            total_chunks += 1

    out_path = Path(args.out_jsonl).resolve()

    print("[stage_2] ---- summary ----")
    print(f"[stage_2] file={src.name}")
    print(f"[stage_2] doc_id={doc_id}")
    print(f"[stage_2] sections={len(sections)} | chunks={total_chunks}")
    print(f"[stage_2] out_jsonl={out_path}")
    print("[stage_2] first_chunk_preview:")
    print(f"[stage_2] entry_date={entry_date} | source_date={source_date}")
    if rows:
        print(rows[0]["text"][:220].replace("\n", "\\n"))

    if args.dry_run:
        print("[stage_2] dry_run=True (no write performed)")
        return

    write_jsonl(out_path, rows)
    print("[stage_2] wrote jsonl")

if __name__ == "__main__":
    main()