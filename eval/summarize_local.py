"""
汇总本地模型对比实验,生成 eval/results/local_compare_report.md 和 eval/results/pass_audit.md。

判分:用 eval/graders.py 的第二版规则直接对已保存的轨迹重新判分(不用重跑模型),
同时保留第一版的结果做对照。
操纵检查:每次运行都检查"思考开/关"是否真的生效;有效运行少于 90% 的配置标为无效,不参与比较。

统计:每题多次运行时,先算每道题的平均成功率,再按题 bootstrap 求置信区间;
两个配置的比较用按题配对 bootstrap 和符号翻转检验(同一道题的多次结果绑在一起)。

事先写下的主要比较:qwen35-nothink vs qwen3-nothink-v2
(原定与 qwen3-nothink 比较,但该配置操纵检查不通过,改用修正后的 v2;此改动在看到 v2 结果之前做出)。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
from graders import final_text, grade_v2, manipulation_check, strip_think  # noqa: E402
from run_eval import load_tasks  # noqa: E402

RESULTS = os.path.join(ROOT, "eval", "results", "local_compare.jsonl")
META = os.path.join(ROOT, "eval", "results", "local_compare_meta.jsonl")
REPORT = os.path.join(ROOT, "eval", "results", "local_compare_report.md")
AUDIT = os.path.join(ROOT, "eval", "results", "pass_audit.md")

ORDER = ["qwen3-think", "qwen3-nothink", "qwen3-nothink-v2", "qwen35-think", "qwen35-nothink"]
B, SEED = 10000, 42
COMPARISONS = [
    ("qwen35-nothink", "qwen3-nothink-v2", True),
    ("qwen3-think", "qwen3-nothink-v2", False),
    ("qwen35-think", "qwen35-nothink", False),
    ("qwen35-think", "qwen3-think", False),
]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_task(rows, field):
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(1.0 if r.get(field) else 0.0)
    return {t: mean(v) for t, v in by.items()}


def boot_ci(vals, rng):
    n = len(vals)
    s = sorted(mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(B))
    return s[int(0.025 * B)], s[int(0.975 * B) - 1]


def sign_flip_p(d, rng):
    obs = abs(mean(d))
    hits = sum(abs(mean(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(B))
    return (hits + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-per-config", type=int, default=3, help="每个配置抽查几条成功样本")
    args = ap.parse_args()
    rng = random.Random(SEED)

    tasks = {t["id"]: t for t in load_tasks(os.path.join(ROOT, "eval", "tasks.jsonl"))}
    rows = load(RESULTS)
    for r in rows:
        r["valid"] = manipulation_check(r)
        r.update(grade_v2(tasks[r["id"]], r) if r["status"] == "ok" else {"success_v2": False})
    meta = {m["config"]: m for m in load(META)}
    configs = sorted({r["config"] for r in rows}, key=lambda c: ORDER.index(c) if c in ORDER else 99)

    L = ["# 本地模型对比报告(判分第二版 + 操纵检查)\n",
         "成功率的置信区间按题 bootstrap(10,000 次);比较用按题配对 bootstrap 和符号翻转检验。"
         "\"有效\"指操纵检查通过(思考开/关真的生效)。\n",
         "## 各配置结果\n",
         "| 配置 | 运行数 | 有效 | 成功率 v1 | 成功率 v2 [95% CI] | 用到正确工具 | 每题耗时 | 每题输出 token | 生成速度 | 首次 prefill | 显存实测增量 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    summ, invalid = {}, set()
    for c in configs:
        rs = [r for r in rows if r["config"] == c]
        ok = [r for r in rs if r["status"] == "ok"]
        valid_rate = mean(1.0 if r["valid"] else 0.0 for r in rs)
        if valid_rate < 0.9:
            invalid.add(c)
        v2 = list(per_task(rs, "success_v2").values())
        lo, hi = boot_ci(v2, rng)
        wall = mean(r["wall_s"] for r in rs)
        dec_tok = sum(r["usage"]["completion_tokens"] for r in ok)
        dec_s = sum(r["usage"]["decode_s"] for r in ok)
        pre = [r["usage"]["first_prefill_s"] for r in ok if r["usage"]["first_prefill_s"] is not None]
        m = meta.get(c, {})
        summ[c] = {"rate": mean(v2), "wall": wall}
        flag = "(**无效**)" if c in invalid else ""
        L.append(f"| {c}{flag} | {len(rs)} | {valid_rate:.0%} | {mean(per_task(rs, 'task_success').values()):.1%} | "
                 f"{mean(v2):.1%} [{lo:.1%}, {hi:.1%}] | {mean(per_task(rs, 'tool_used').values()):.1%} | "
                 f"{wall:.1f} s | {mean(r['usage']['completion_tokens'] for r in ok):.0f} | "
                 f"{dec_tok / dec_s if dec_s else float('nan'):.1f} tok/s | "
                 f"{mean(pre) if pre else float('nan'):.2f} s | {m.get('vram_delta_mib', '—')} MiB |")

    L += ["\n## v1 → v2 判分变化\n", "| 配置 | v1 判错、v2 判对 | v1 判对、v2 判错 |", "|---|---|---|"]
    for c in configs:
        rs = [r for r in rows if r["config"] == c]
        up = [f"{r['id']}#{r['run']}" for r in rs if r.get("success_v2") and not r.get("task_success")]
        down = [f"{r['id']}#{r['run']}" for r in rs if r.get("task_success") and not r.get("success_v2")]
        L.append(f"| {c} | {len(up)}:{' '.join(up) or '—'} | {len(down)}:{' '.join(down) or '—'} |")

    L += ["\n## 配对比较(只用有效配置,判分第二版)\n",
          "| 比较 | 成功率差 [95% CI] | 符号翻转 p | 耗时比 | 说明 |", "|---|---|---|---|---|"]
    for a, b, primary in COMPARISONS:
        if a not in summ or b not in summ:
            L.append(f"| {a} − {b} | — | — | — | 缺少数据 |")
            continue
        if a in invalid or b in invalid:
            L.append(f"| {a} − {b} | — | — | — | 有无效配置,不比较 |")
            continue
        ra = per_task([r for r in rows if r["config"] == a], "success_v2")
        rb = per_task([r for r in rows if r["config"] == b], "success_v2")
        d = [ra[t] - rb[t] for t in sorted(set(ra) & set(rb))]
        lo, hi = boot_ci(d, rng)
        L.append(f"| {a} − {b} | {mean(d):+.1%} [{lo:+.1%}, {hi:+.1%}] | {sign_flip_p(d, rng):.3f} | "
                 f"{summ[a]['wall'] / summ[b]['wall']:.1f}× | {'**主要比较**' if primary else '探索性'} |")

    L += ["\n## 判分器精确率诊断(在 v2 判为成功的运行里)\n",
          "| 配置 | 计算题:回答里没有出现正确数值 | 概念题:回答没提到该术语 | 概念题:还查到了不相关的词条 |",
          "|---|---|---|---|"]
    for c in configs:
        ps = [r for r in rows if r["config"] == c and r.get("success_v2")]
        num = [r for r in ps if r["check_type"] == "numeric"]
        term = [r for r in ps if r["check_type"] == "term"]
        L.append(f"| {c} | {sum(not r.get('answer_has_value') for r in num)}/{len(num)} | "
                 f"{sum(not r.get('answer_mentions_term') for r in term)}/{len(term)} | "
                 f"{sum(bool(r.get('wrong_lookup')) for r in term)}/{len(term)} |")

    L += ["\n## 注意\n",
          "- 只有 12 道题,置信区间很宽,结论是初步观察;扩大评测集后再下定论。",
          "- 比较的是各模型\"推荐采样设置下\"的表现。",
          "- \"如实报告失败\"类目前仍用关键词判断,计划换成 LLM-as-Judge。",
          "- 判分器精确率诊断只是线索,需要配合 pass_audit.md 人工抽查。"]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- 抽查清单:每个配置随机抽几条 v2 判为成功的运行,供人工判断是否真的对 ----
    A = ["# 成功样本抽查清单\n", "逐条判断:最终回答是否真的正确、是否用到了工具结果、有没有编造。"
         "在每条下面写下你的判断(对 / 错 / 部分对 + 理由)。\n"]
    for c in configs:
        ps = [r for r in rows if r["config"] == c and r.get("success_v2")]
        for r in random.Random(SEED).sample(ps, min(args.audit_per_config, len(ps))):
            t = tasks[r["id"]]
            A += [f"## {c} · {r['id']} · run{r['run']}\n", f"**题目**:{t['task']}\n",
                  f"**工具调用**:{', '.join(f'{p}' for p in [s['content']['tool'] + str(s['content']['input']) for s in r['trajectory']['steps'] if s['type'] == 'tool_call']) or '无'}\n",
                  "**最终回答**:\n", "> " + strip_think(final_text(r["trajectory"]))[:800].replace("\n", "\n> ") + "\n",
                  "**你的判断**:\n"]
    with open(AUDIT, "w", encoding="utf-8") as f:
        f.write("\n".join(A) + "\n")
    print(f"\n报告:{REPORT}\n抽查清单:{AUDIT}")


if __name__ == "__main__":
    main()
