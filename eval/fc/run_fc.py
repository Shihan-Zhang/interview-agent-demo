"""
在 BFCL 子集上评测本地模型的函数调用能力。每题只调用一次模型,不执行函数,只判调用对不对。

配置沿用 compare_local.py 里操纵检查通过的 4 个。默认每题跑 1 次:
每题一个"对/错"结果,配置之间正好用精确 McNemar 检验比较。

运行(项目根目录):
    python eval/fc/run_fc.py                      # 4 个配置 × 40 题
    python eval/fc/run_fc.py --configs qwen35-nothink --limit 5   # 先试 5 题
    python eval/fc/summarize_fc.py
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, HERE)

from compare_local import COMMON, CONFIGS, unload_all, warm_up  # noqa: E402
from fc_grader import grade  # noqa: E402
from providers.ollama_provider import DEFAULT_BASE_URL, parse_response  # noqa: E402

VALID_CONFIGS = ["qwen3-think", "qwen3i-nothink", "qwen35-think", "qwen35-nothink"]
DATA = os.path.join(ROOT, "eval", "datasets", "bfcl_subset_v1.jsonl")
RESULTS = os.path.join(ROOT, "eval", "results", "fc_bfcl.jsonl")


def chat(model, think, options, messages, tools):
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "think": think,
                       "stream": False, "options": options, "keep_alive": "30m"}).encode("utf-8")
    req = urllib.request.Request(f"{DEFAULT_BASE_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(VALID_CONFIGS))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题,用来试跑")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    with open(DATA, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()][: args.limit]
    names = args.configs.split(",")
    configs = [c for c in CONFIGS if c["name"] in names]
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            done = {(r["config"], r["id"], r["run"]) for r in map(json.loads, f) if r["status"] == "ok"}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with urllib.request.urlopen(f"{DEFAULT_BASE_URL}/api/version", timeout=10) as resp:
        ollama_version = json.loads(resp.read().decode("utf-8")).get("version")
    run_id = time.strftime("%Y%m%d-%H%M%S")

    for cfg in configs:
        todo = [(run, i, it) for run in range(1, args.runs + 1) for i, it in enumerate(items)
                if (cfg["name"], it["id"], run) not in done]
        if not todo:
            continue
        unload_all()
        warm_up(cfg["model"], cfg["think"])
        print(f"\n[{cfg['name']}] {len(todo)} 题")
        for run, i, it in todo:
            options = {**COMMON, **cfg["sampling"], "seed": run * 100 + i}
            row = {"config": cfg["name"], "model": cfg["model"], "think": cfg["think"], "id": it["id"],
                   "category": it["category"], "run": run, "options": options,
                   "run_id": run_id, "base_url": DEFAULT_BASE_URL, "ollama_version": ollama_version}  # 数据血缘
            t0 = time.time()
            try:
                data = chat(cfg["model"], cfg["think"], options, it["messages"], it["tools"])
                d = parse_response(data, 0)
                calls = [(it["name_map"].get(c["tool"], c["tool"]), c["input"]) for c in d.get("calls", [])]
                ok, reason = grade(it, calls)
                row.update({"status": "ok", "success": ok, "reason": reason, "calls": calls,
                            "content": data.get("message", {}).get("content", ""), "usage": d["usage"]})
            except Exception as e:
                row.update({"status": "error", "success": False, "reason": repr(e)})
            row["wall_s"] = time.time() - t0
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{cfg['name']}] {it['id']:<22} {'PASS' if row['success'] else 'FAIL'} "
                  f"{row['wall_s']:5.1f}s  {'' if row['success'] else row['reason'][:70]}")

    print(f"\n完成。结果在 {RESULTS},运行 python eval/fc/summarize_fc.py 生成报告。")


if __name__ == "__main__":
    main()
