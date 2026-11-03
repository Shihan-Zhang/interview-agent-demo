"""
汇总 BFCL 子集的结果,生成 eval/results/fc_<数据集名>_report.md。

每题每配置一个"对/错"结果(第 1 次运行),所以:
- 单个配置的准确率用 Wilson 置信区间(题少时比正态近似可靠);
- 两个配置的比较用精确 McNemar 检验:只看两者结果不一致的题,
  在"两者一样好"的原假设下,"只有 A 对"的题数服从 Binomial(不一致题数, 1/2)。
  比较只在两个配置都跑过的题上做,题数写在表里。

判分和运行分开:这里用当前的 fc_grader 对存下来的调用重新判分,
判分规则修正后不需要重跑模型。运行时记录的判分结果只作对照。

数据质量检查(任何一项非零都要先查清楚再看结论):
- 来源可疑:耗时 < 0.05 s(真实调用至少约 1 秒,见 2026-11-03 假数据事故);
- 可能被截断:输入 token 数达到 num_ctx 的 95%(Ollama 超出上下文时静默截断);
- 输出被截断:done_reason == "length"(达到 num_predict 上限)。

用法:python eval/fc/summarize_fc.py [--data eval/datasets/bfcl_subset_v2.jsonl]
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fc_grader import grade  # noqa: E402

DATA = os.path.join(ROOT, "eval", "datasets", "bfcl_subset_v2.jsonl")
RESULTS = os.path.join(ROOT, "eval", "results", "fc_bfcl.jsonl")
ORDER = ["qwen3-think", "qwen3i-nothink", "qwen35-think", "qwen35-nothink"]
COMPARISONS = [("qwen35-nothink", "qwen3i-nothink", True), ("qwen35-think", "qwen35-nothink", False),
               ("qwen3-think", "qwen3i-nothink", False), ("qwen35-think", "qwen3-think", False)]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def valid(r):
    leaked = "<think>" in (r.get("content") or "") or "</think>" in (r.get("content") or "")
    thinking = (r.get("usage") or {}).get("thinking_chars", 0) or 0
    return r["status"] == "ok" and not leaked and (thinking > 0 if r["think"] else thinking == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    name = os.path.splitext(os.path.basename(args.data))[0].replace("_subset", "")
    report = os.path.join(ROOT, "eval", "results", f"fc_{name}_report.md")

    with open(args.data, encoding="utf-8") as f:
        items = {d["id"]: d for d in map(json.loads, f)}
    with open(RESULTS, encoding="utf-8") as f:
        rows = [r for r in (json.loads(line) for line in f if line.strip()) if r["id"] in items]

    changed = suspicious = truncated_in = truncated_out = 0
    for r in rows:
        if r["status"] == "ok":
            ok, reason = grade(items[r["id"]], [tuple(c) for c in r["calls"]])
            changed += ok != r["success"]
            r["success"], r["reason"] = ok, reason
            u = r.get("usage") or {}
            if u.get("prompt_tokens", 0) >= 0.95 * r["options"]["num_ctx"]:
                truncated_in += 1
            if u.get("done_reason") == "length":
                truncated_out += 1
        if r.get("wall_s", 1) < 0.05:
            suspicious += 1

    first = {(r["config"], r["id"]): r for r in rows if r["run"] == 1}
    configs = sorted({c for c, _ in first}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    cats = list(dict.fromkeys(d["category"] for d in items.values()))

    L = [f"# BFCL 子集函数调用评测({name})\n",
         f"数据集 `{os.path.basename(args.data)}`,{len(items)} 题,BFCL v4 分层抽样(Apache-2.0)。"
         "每题每配置 1 次,准确率用 Wilson 95% 置信区间,比较用精确 McNemar 检验。\n",
         "## 数据质量检查\n",
         f"- 重新判分改变的行:{changed}",
         f"- 来源可疑(耗时 < 0.05 s):{suspicious}",
         f"- 输入可能被截断(≥ 95% num_ctx):{truncated_in}",
         f"- 输出被截断(done_reason = length):{truncated_out}\n",
         "## 各配置总体结果\n",
         "| 配置 | 完成题数 | 有效 | 准确率 [95% CI] | 每题耗时 | 每题输出 token |",
         "|---|---|---|---|---|---|"]
    by_cfg = {c: [r for (cc, _), r in first.items() if cc == c] for c in configs}
    for c in configs:
        rs = by_cfg[c]
        k, n = sum(r["success"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        ok = [r for r in rs if r["status"] == "ok"]
        L.append(f"| {c} | {n}/{len(items)} | {sum(valid(r) for r in rs)}/{n} | {k}/{n} = {k / n:.1%} "
                 f"[{lo:.1%}, {hi:.1%}] | {sum(r['wall_s'] for r in rs) / n:.1f} s | "
                 f"{sum(r['usage']['completion_tokens'] for r in ok) / max(len(ok), 1):.0f} |")

    L += ["\n## 分类别准确率\n", "| 类别 | 题数 | " + " | ".join(configs) + " |",
          "|---|---|" + "---|" * len(configs)]
    for ct in cats:
        n_ct = sum(d["category"] == ct for d in items.values())
        cells = []
        for c in configs:
            rs = [r for r in by_cfg[c] if r["category"] == ct]
            cells.append(f"{sum(r['success'] for r in rs)}/{len(rs)}" if rs else "—")
        L.append(f"| {ct} | {n_ct} | " + " | ".join(cells) + " |")

    L += ["\n## 配对比较(精确 McNemar)\n",
          "| 比较 | 共同题数 | 都对 | 只有 A 对 (b) | 只有 B 对 (c) | 都错 | 准确率差 | p 值 | 说明 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, b_, primary in COMPARISONS:
        ids = sorted(i for i in items if (a, i) in first and (b_, i) in first)
        if not ids:
            continue
        pa = [first[(a, i)]["success"] for i in ids]
        pb = [first[(b_, i)]["success"] for i in ids]
        both = sum(x and y for x, y in zip(pa, pb))
        only_a = sum(x and not y for x, y in zip(pa, pb))
        only_b = sum(y and not x for x, y in zip(pa, pb))
        neither = len(ids) - both - only_a - only_b
        L.append(f"| {a} vs {b_} | {len(ids)} | {both} | {only_a} | {only_b} | {neither} | "
                 f"{(only_a - only_b) / len(ids):+.1%} | {mcnemar_exact(only_a, only_b):.3f} | "
                 f"{'**主要比较**' if primary else '探索性'} |")

    L += ["\n## 失败明细(用于 Trace 分析)\n", "| 配置 | 题 | 类别 | 原因 |", "|---|---|---|---|"]
    for (c, i), r in sorted(first.items(), key=lambda kv: (ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 99, kv[0][1])):
        if not r["success"]:
            L.append(f"| {c} | {i} | {r['category']} | {r['reason'][:120].replace('|', '/')} |")

    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n报告:{report}")


if __name__ == "__main__":
    main()
