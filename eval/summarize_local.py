"""
汇总本地模型对比实验的结果,生成 eval/results/local_compare_report.md。

统计方法(和单题只跑一次时用的 McNemar 不同,这里每题跑了多次):
- 每个配置的成功率:先算每道题在多次运行里的平均成功率,再按题做 bootstrap 求置信区间
  (按题重抽,同一道题的多次结果绑在一起,不当成独立样本);
- 两个配置的比较:每道题的成功率差 d_i,用配对 bootstrap 求差值的置信区间,
  用符号翻转(置换)检验求 p 值。

事先写下的主要比较和决策规则(在看结果之前定好):
- 主要比较:qwen35-nothink vs qwen3-nothink;
- 思考模式:如果开思考的成功率提升不显著,且平均每题耗时增加超过 2 倍,就默认关闭思考。
"""

import json
import os
import random
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval", "results", "local_compare.jsonl")
META = os.path.join(ROOT, "eval", "results", "local_compare_meta.jsonl")
REPORT = os.path.join(ROOT, "eval", "results", "local_compare_report.md")

B = 10000
SEED = 42
COMPARISONS = [  # (A, B, 是否为事先定好的主要比较)
    ("qwen35-nothink", "qwen3-nothink", True),
    ("qwen3-think", "qwen3-nothink", False),
    ("qwen35-think", "qwen35-nothink", False),
    ("qwen35-think", "qwen3-think", False),
]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_task_rate(rows, config, field="task_success"):
    by_task = defaultdict(list)
    for r in rows:
        if r["config"] == config:
            by_task[r["id"]].append(1.0 if r.get(field) else 0.0)
    return {t: mean(v) for t, v in by_task.items()}


def bootstrap_ci(values, rng):
    n = len(values)
    stats = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(B))
    return stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def sign_flip_p(diffs, rng):
    obs = abs(mean(diffs))
    hits = sum(abs(mean(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(B))
    return (hits + 1) / (B + 1)


def main():
    rows = [r for r in load(RESULTS)]
    meta = {m["config"]: m for m in load(META)}  # 同一配置多次测量时,取最后一次
    rng = random.Random(SEED)
    configs = sorted({r["config"] for r in rows}, key=lambda c: ["qwen3-think", "qwen3-nothink", "qwen35-think", "qwen35-nothink"].index(c) if c in ["qwen3-think", "qwen3-nothink", "qwen35-think", "qwen35-nothink"] else 99)
    L = ["# 本地模型对比报告\n",
         "每个配置:12 道题 × 多次运行。成功率的置信区间按题 bootstrap(10,000 次);"
         "比较用按题配对 bootstrap 和符号翻转检验。\n",
         "## 各配置结果\n",
         "| 配置 | 运行数 | 任务成功率 [95% CI] | 工具选对率 | 每题平均耗时 | 每题输出 token | 生成速度 | 首次 prefill | 截断 | 出错 | 显存(Ollama 报告) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for c in configs:
        rs = [r for r in rows if r["config"] == c]
        ok = [r for r in rs if r["status"] == "ok"]
        rates = list(per_task_rate(rows, c).values())
        lo, hi = bootstrap_ci(rates, rng)
        tool = mean(per_task_rate(rows, c, "tool_correct").values())
        wall = mean(r["wall_s"] for r in rs)
        out_tok = mean(r["usage"]["completion_tokens"] for r in ok) if ok else float("nan")
        dec_tok = sum(r["usage"]["completion_tokens"] for r in ok)
        dec_s = sum(r["usage"]["decode_s"] for r in ok)
        prefill = [r["usage"]["first_prefill_s"] for r in ok if r["usage"]["first_prefill_s"] is not None]
        m = meta.get(c, {})
        vram = f"{m['ollama_size'] / 1e9:.2f} GB({m['ollama_size_vram'] / m['ollama_size']:.0%} GPU)" \
            if m.get("ollama_size") else "—"
        summary[c] = {"rate": mean(rates), "wall": wall}
        L.append(f"| {c} | {len(rs)} | {mean(rates):.1%} [{lo:.1%}, {hi:.1%}] | {tool:.1%} | {wall:.1f} s | "
                 f"{out_tok:.0f} | {dec_tok / dec_s if dec_s else float('nan'):.1f} tok/s | "
                 f"{mean(prefill) if prefill else float('nan'):.2f} s | "
                 f"{sum(1 for r in ok if r['usage']['truncated'])} | {len(rs) - len(ok)} | {vram} |")

    L += ["\n## 配对比较\n",
          "| 比较 | 成功率差 [95% CI] | 符号翻转 p | 耗时比 | 说明 |", "|---|---|---|---|---|"]
    for a, b, primary in COMPARISONS:
        if a not in summary or b not in summary:
            continue
        ra, rb = per_task_rate(rows, a), per_task_rate(rows, b)
        common = sorted(set(ra) & set(rb))
        d = [ra[t] - rb[t] for t in common]
        lo, hi = bootstrap_ci(d, rng)
        p = sign_flip_p(d, rng)
        ratio = summary[a]["wall"] / summary[b]["wall"]
        L.append(f"| {a} − {b} | {mean(d):+.1%} [{lo:+.1%}, {hi:+.1%}] | {p:.3f} | {ratio:.1f}× | "
                 f"{'**主要比较**' if primary else '探索性'} |")

    L += ["\n## 思考模式的决策(按事先定好的规则)\n"]
    for model, on, off in (("qwen3:4b", "qwen3-think", "qwen3-nothink"), ("qwen3.5:4b", "qwen35-think", "qwen35-nothink")):
        if on in summary and off in summary:
            ra, rb = per_task_rate(rows, on), per_task_rate(rows, off)
            d = [ra[t] - rb[t] for t in sorted(set(ra) & set(rb))]
            lo, _ = bootstrap_ci(d, rng)
            ratio = summary[on]["wall"] / summary[off]["wall"]
            sig_better = lo > 0
            decision = "默认关闭思考" if (not sig_better and ratio > 2) else "需要人工判断"
            L.append(f"- {model}:开思考成功率差 {mean(d):+.1%}(显著更好:{'是' if sig_better else '否'}),"
                     f"耗时 {ratio:.1f} 倍 → **{decision}**")

    L += ["\n## 按题型\n", "| 配置 | " + " | ".join(sorted({r['check_type'] for r in rows})) + " |",
          "|---|" + "---|" * len({r['check_type'] for r in rows})]
    for c in configs:
        cells = []
        for ct in sorted({r["check_type"] for r in rows}):
            v = [1.0 if r.get("task_success") else 0.0 for r in rows if r["config"] == c and r["check_type"] == ct]
            cells.append(f"{mean(v):.0%}(n={len(v)})" if v else "—")
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    L += ["\n## 注意\n",
          "- 只有 12 道题,置信区间很宽,以上结论只是初步观察;扩大评测集后再下定论。",
          "- 采样参数用的是各模型的推荐设置,所以比较的是\"各自推荐设置下\"的表现,不是同等条件下的模型本身。",
          "- 生成速度 = 总输出 token / 总 decode 时间;首次 prefill 近似第一步的 TTFT(本地单请求,没有排队)。"]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n报告已写入 {REPORT}")


if __name__ == "__main__":
    main()
