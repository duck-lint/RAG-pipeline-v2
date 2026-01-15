python -c "import json
p='stage_2_chunks.jsonl'
with open(p,'r',encoding='utf-8') as f:
  for i,line in enumerate(f):
    row=json.loads(line)
    print('--- chunk', i, '---')
    print('anchor:', row['metadata'].get('chunk_anchor'), 'len:', len(row['text']))
    print(row['text'][:400])
    print()"
