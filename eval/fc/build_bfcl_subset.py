"""
从 BFCL v4(Berkeley Function Calling Leaderboard,Apache-2.0)分层抽样函数调用评测子集。

为什么抽样而不是全用:本地 4B 模型跑全部 5,000 多题要好几个小时;按类别分层抽样,
每个类别都有代表,固定种子可复现。

版本:
- v1(40 题):simple_python 15、multiple 10、parallel 5、irrelevance 10。
  结果:4 个配置都在 85%–97.5%,irrelevance 几乎满分(天花板效应),且 40 题只能检出约 14 个百分点的差距。
- v2(150 题)= v1 原样(顺序不变,已有结果和随机种子可以复用)+ 110 道更难的题:
  live_* 类别来自真实用户提交的问题和函数,描述更乱、函数更多(live_multiple 最多 37 个),
  live_irrelevance 用来解决 irrelevance 的天花板;live_relevance 要求"至少调用一个函数"。
  过滤:题目加函数定义超过 16,000 字符(约 5,000 token)的题不入选,保证在 num_ctx=8192 内不被截断;
  全部 live 题中被过滤掉的只有个位数,对代表性影响很小,但要在报告里说明。

转换:
- BFCL 的参数类型用 Python 写法(dict / float / tuple / any),Ollama 需要标准 JSON Schema
  (object / number / array / string),这里递归转换;
- 函数名里的点号(如 math.factorial)不符合多数接口对函数名的要求,和 BFCL 官方做法一样换成下划线,
  同时保存原名,判分时换回来。

用法:python eval/fc/build_bfcl_subset.py <BFCL 的 bfcl_eval/data 目录> [v1|v2]
"""

import json
import os
import random
import sys

SAMPLE_V1 = {"simple_python": 15, "multiple": 10, "parallel": 5, "irrelevance": 10}
SEED_V1 = 20261103
SAMPLE_V2_EXTRA = {"live_simple": 30, "live_multiple": 30, "live_parallel": 5, "live_parallel_multiple": 5,
                   "parallel_multiple": 10, "live_irrelevance": 25, "live_relevance": 5}
SEED_V2 = 20261104
MAX_CHARS = 16000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
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


def size(item):
    return len(json.dumps(item["function"])) + len(json.dumps(item["question"]))


def sample(data_dir, spec, seed, max_chars=None):
    rng = random.Random(seed)
    out = []
    for cat, n in spec.items():
        items = load(os.path.join(data_dir, f"BFCL_v4_{cat}.json"))
        if max_chars:
            items = [it for it in items if size(it) <= max_chars]   # 先过滤再抽样,保证可复现
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
                "ground_truth": answers.get(item["id"], []),   # irrelevance / relevance 没有答案文件
            })
    return out


def main(data_dir, version="v2"):
    rows = sample(data_dir, SAMPLE_V1, SEED_V1)
    if version == "v2":
        rows += sample(data_dir, SAMPLE_V2_EXTRA, SEED_V2, MAX_CHARS)
    out = os.path.normpath(os.path.join(OUT_DIR, f"bfcl_subset_{version}.jsonl"))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(rows)} 题 -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v2")
