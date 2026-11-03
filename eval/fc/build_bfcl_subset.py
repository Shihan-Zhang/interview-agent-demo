"""
从 BFCL v4(Berkeley Function Calling Leaderboard,Apache-2.0)分层抽样一个函数调用评测子集。

为什么抽样而不是全用:本地 4B 模型跑全部 5,000 多题要好几个小时;按类别分层抽 40 题,
足够看出明显差别,也能保证每个类别都有代表。抽样用固定种子,可复现。

转换:
- BFCL 的参数类型用 Python 写法(dict / float / tuple / any),Ollama 需要标准 JSON Schema
  (object / number / array / string),这里递归转换;
- 函数名里的点号(如 math.factorial)不符合多数接口对函数名的要求,和 BFCL 官方做法一样换成下划线,
  同时保存原名,判分时换回来。

用法:python eval/fc/build_bfcl_subset.py <BFCL 的 bfcl_eval/data 目录>
"""

import json
import os
import random
import sys

SAMPLE = {"simple_python": 15, "multiple": 10, "parallel": 5, "irrelevance": 10}
SEED = 20261103
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "bfcl_subset_v1.jsonl")
TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def convert_schema(x):
    if isinstance(x, dict):
        y = {k: convert_schema(v) for k, v in x.items()}
        if isinstance(y.get("type"), str):
            y["type"] = TYPE_MAP.get(y["type"], y["type"])
        return y
    if isinstance(x, list):
        return [convert_schema(v) for v in x]
    return x


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(data_dir):
    rng = random.Random(SEED)
    out = []
    for cat, n in SAMPLE.items():
        items = load(os.path.join(data_dir, f"BFCL_v4_{cat}.json"))
        answers = {}
        ans_path = os.path.join(data_dir, "possible_answer", f"BFCL_v4_{cat}.json")
        if os.path.exists(ans_path):
            answers = {a["id"]: a["ground_truth"] for a in load(ans_path)}
        for item in sorted(rng.sample(items, n), key=lambda d: d["id"]):
            tools, name_map = [], {}
            for fn in item["function"]:
                safe = fn["name"].replace(".", "_")
                name_map[safe] = fn["name"]
                tools.append({"type": "function", "function": {
                    "name": safe, "description": fn.get("description", ""),
                    "parameters": convert_schema(fn["parameters"])}})
            out.append({
                "id": item["id"], "source": "BFCL_v4", "license": "Apache-2.0", "category": cat,
                "messages": item["question"][0], "tools": tools, "name_map": name_map,
                "ground_truth": answers.get(item["id"], []),   # irrelevance:不应调用任何函数,没有答案文件
            })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(out)} 题 -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main(sys.argv[1])
