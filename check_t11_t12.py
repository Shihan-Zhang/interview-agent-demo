import json

with open('eval/eval_results.jsonl', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row['id'] in ('t11', 't12'):
            print(f"===== {row['id']} =====")
            print('task:', row['task'])
            for step in row['trajectory']['steps']:
                content_str = json.dumps(step['content'], ensure_ascii=False)
                print(f"  [{step['type']}] {content_str[:300]}")
            print()