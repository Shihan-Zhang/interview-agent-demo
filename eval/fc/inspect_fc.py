"""
排查 BFCL 结果里的数据质量问题,打印需要人工看的行。

    python eval/fc/inspect_fc.py              # 列出报错行、输出被截断的行
    python eval/fc/inspect_fc.py <题目 id>     # 打印这道题的问题、标准答案和各配置的调用(Trace 分析用)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "eval", "results", "fc_bfcl.jsonl")
DATA = os.path.join(ROOT, "eval", "datasets", "bfcl_subset_v2.jsonl")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    with open(RESULTS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(sys.argv) > 1:
        with open(DATA, encoding="utf-8") as f:
            item = next(d for d in map(json.loads, f) if d["id"] == sys.argv[1])
        print("问题:", json.dumps(item["messages"], ensure_ascii=False, indent=1))
        print("函数:", [t["function"]["name"] for t in item["tools"]])
        print("标准答案:", json.dumps(item["ground_truth"], ensure_ascii=False))
        for r in rows:
            if r["id"] == sys.argv[1]:
                print(f"\n[{r['config']}] status={r['status']} calls={json.dumps(r.get('calls'), ensure_ascii=False)}")
                if r.get("content"):
                    print("  正文:", r["content"][:300])
        return
    for r in rows:
        u = r.get("usage") or {}
        if r["status"] != "ok":
            print(f"报错    {r['config']:<16} {r['id']:<30} {r['reason'][:80]}")
        elif u.get("done_reason") == "length":
            print(f"输出截断 {r['config']:<16} {r['id']:<30} 输出 {u.get('completion_tokens')} token,"
                  f"思考 {u.get('thinking_chars')} 字符,calls={len(r.get('calls') or [])}")


if __name__ == "__main__":
    main()
