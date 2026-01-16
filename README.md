# v2pipeline Usage

## Stage 3 ingestion modes

Rebuild (delete and recreate the collection; optionally reset the persist dir):
```bash
python 03_stage3_build_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode rebuild --reset_db
```

Append (add only; fails if any ids already exist):
```bash
python 03_stage3_build_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode append
```

Upsert (default; overwrite existing ids when supported):
```bash
python 03_stage3_build_chroma.py --chunks_jsonl stage_2_chunks_merged.jsonl --mode upsert
```
