"""
汇总 BFCL 子集的结果,生成 eval/results/fc_bfcl_report.md。

每题每配置一个"对/错"结果(第 1 次运行),所以:
- 单个配置的准确率用 Wilson 置信区间(题少时比正态近似可靠);
- 两个配置的比较用精确 McNemar 检验:只看两者结果不一致的题,
  在"两者一样好"的原假设下,"只有 A 对"的题数服从 Binomial(不一致题数, 1/2)。
"""

import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "eval", "results", "fc_bfcl.jsonl")
REPORT = os.path.join(ROOT, "eval", "results", "fc_bfcl_report.md")
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
    with open(RESULTS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    first = {(r["config"], r["id"]): r for r in rows if r["run"] == 1}
    configs = sorted({r["config"] for r in rows}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    cats = sorted({r["category"] for r in rows})

    L = ["# BFCL 子集函数调用评测\n",
         f"BFCL v4 分层抽样 40 题(Apache-2.0)。每题每配置 1 次,准确率用 Wilson 95% 置信区间,比较用精确 McNemar 检验。\n",
         "## 各配置结果\n",
         "| 配置 | 有效 | 准确率 [95% CI] | " + " | ".join(cats) + " | 每题耗时 | 每题输出 token |",
         "|---|---|---|" + "---|" * len(cats) + "---|---|"]
    for c in configs:
        rs = [r for (cc, _), r in first.items() if cc == c]
        k, n = sum(r["success"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        by = defaultdict(list)
        for r in rs:
            by[r["category"]].append(r["success"])
        ok = [r for r in rs if r["status"] == "ok"]
        L.append(f"| {c} | {sum(valid(r) for r in rs)}/{n} | {k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}] | "
                 + " | ".join(f"{sum(by[ct])}/{len(by[ct])}" for ct in cats)
                 + f" | {sum(r['wall_s'] for r in rs) / n:.1f} s | "
                 f"{sum(r['usage']['completion_tokens'] for r in ok) / max(len(ok), 1):.0f} |")

    L += ["\n## 配对比较(精确 McNemar)\n",
          "| 比较 | 都对 | 只有 A 对 (b) | 只有 B 对 (c) | 都错 | 准确率差 | p 值 | 说明 |",
          "|---|---|---|---|---|---|---|---|"]
    ids = sorted({i for _, i in first})
    for a, b_, primary in COMPARISONS:
        if not all((x, i) in first for x in (a, b_) for i in ids):
            L.append(f"| {a} vs {b_} | — | — | — | — | — | — | 缺少数据 |")
            continue
        pa = [first[(a, i)]["success"] for i in ids]
        pb = [first[(b_, i)]["success"] for i in ids]
        both = sum(x and y for x, y in zip(pa, pb))
        only_a = sum(x and not y for x, y in zip(pa, pb))
        only_b = sum(y and not x for x, y in zip(pa, pb))
        neither = len(ids) - both - only_a - only_b
        L.append(f"| {a} vs {b_} | {both} | {only_a} | {only_b} | {neither} | "
                 f"{(only_a - only_b) / len(ids):+.1%} | {mcnemar_exact(only_a, only_b):.3f} | "
                 f"{'**主要比较**' if primary else '探索性'} |")

    L += ["\n## 失败明细(用于 Trace 分析)\n", "| 配置 | 题 | 类别 | 原因 |", "|---|---|---|---|"]
    for (c, i), r in sorted(first.items(), key=lambda kv: (ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 99, kv[0][1])):
        if not r["success"]:
            L.append(f"| {c} | {i} | {r['category']} | {r['reason'][:120].replace('|', '/')} |")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n报告:{REPORT}")


if __name__ == "__main__":
    main()
