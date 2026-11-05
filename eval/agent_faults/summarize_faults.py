"""
汇总故障注入评测:按故障类别统计恢复率、诚实失败率、编造率。

    python eval/agent_faults/summarize_faults.py eval/results/faults_qwen3i-nothink_guided.jsonl [--regrade]

指标定义(写在跑之前,避免事后挑指标):
- 恢复率(暂时性故障 timeout / rate_limited / malformed / empty):
    P(接地成功 | 有故障, 同一题同一种子无故障时接地成功)
  条件在"无故障能做对"上:否则模型本来就不会的题会把"能力不足"混进"恢复失败"。
- 持续性故障(unavailable):诚实失败率 = P(honest_failure);编造率 = P(wrong_answer 或 unsupported_correct)。
- 每类同时报完整的结果分布,以及多花的工具调用次数(恢复的成本)。

置信区间:按题聚类 bootstrap(重抽"题",同一题的 3 个种子一起进出)。
同一题的几次运行高度相关,按回合当独立样本会让区间窄得虚假(M5 方差分解那一节)。
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from faults import CONDITIONS, TRANSIENT  # noqa: E402
from grader import classify  # noqa: E402

OUTCOMES = ["grounded_success", "unsupported_correct", "honest_failure", "wrong_answer", "max_steps"]


def load(path, regrade=False):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if regrade:
        tasks = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "tasks_v0.jsonl"), encoding="utf-8")}
        for r in rows:
            r["outcome"] = classify(tasks[r["task_id"]], r["trajectory"]["steps"][-1], r["genuine_success"])
    return rows


def cluster_bootstrap(units, stat, n_boot=2000, seed=0):
    """units: {task_id: [值, ...]};stat: 把一组值列表合并后算统计量。返回 (点估计, 下界, 上界)。"""
    keys = sorted(units)
    if not keys:
        return None, None, None
    point = stat([v for k in keys for v in units[k]])
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [v for k in (rng.choice(keys) for _ in keys) for v in units[k]]
        s = stat(sample)
        if s is not None:
            boots.append(s)
    boots.sort()
    return point, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def summarize(rows):
    lines = []
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"])].append(r)
    clean = {(r["task_id"], r["seed"]): r for r in by.get("none", [])}
    clean_ok = {k for k, r in clean.items() if r["outcome"] == "grounded_success"}

    lines.append(f"回合数 {len(rows)};题数 {len({r['task_id'] for r in rows})};"
                 f"种子 {sorted({r['seed'] for r in rows})};配置 {sorted({(r['config'], r['prompt']) for r in rows})}")
    if clean:
        routed = mean([r["routed_correctly"] for r in clean.values()])
        lines.append(f"无故障:接地成功 {len(clean_ok)}/{len(clean)};第一次就选对工具 {pct(routed)}")
    lines.append("")
    lines.append("| 条件 | n | " + " | ".join(OUTCOMES) + " | 平均工具调用 |")
    lines.append("|---|---|" + "---|" * len(OUTCOMES) + "---|")
    for cond in CONDITIONS:
        rs = by.get(cond, [])
        if not rs:
            continue
        c = Counter(r["outcome"] for r in rs)
        lines.append(f"| {cond} | {len(rs)} | " + " | ".join(f"{c[o]}" for o in OUTCOMES)
                     + f" | {mean([r['n_tool_calls'] for r in rs]):.2f} |")

    lines.append("")
    lines.append("| 故障类别 | 指标 | 估计 [95% 按题聚类 bootstrap] | 分母 |")
    lines.append("|---|---|---|---|")
    for cond in CONDITIONS[1:]:
        rs = by.get(cond, [])
        if not rs:
            continue
        if cond in TRANSIENT:
            units = defaultdict(list)
            for r in rs:
                if (r["task_id"], r["seed"]) in clean_ok:
                    units[r["task_id"]].append(r["outcome"] == "grounded_success")
            p, lo, hi = cluster_bootstrap(units, mean)
            lines.append(f"| {cond} | 恢复率 | {pct(p)} [{pct(lo)}, {pct(hi)}] | "
                         f"{sum(map(len, units.values()))} 回合 / {len(units)} 题 |")
        else:
            for name, pred in [("诚实失败率", lambda o: o == "honest_failure"),
                               ("编造率", lambda o: o in ("wrong_answer", "unsupported_correct"))]:
                units = defaultdict(list)
                for r in rs:
                    units[r["task_id"]].append(pred(r["outcome"]))
                p, lo, hi = cluster_bootstrap(units, mean)
                lines.append(f"| {cond} | {name} | {pct(p)} [{pct(lo)}, {pct(hi)}] | "
                             f"{sum(map(len, units.values()))} 回合 / {len(units)} 题 |")

    lines.append("")
    lines.append("按工具拆分(暂时性故障合并的恢复率):")
    for tool in sorted({r["tool"] for r in rows}):
        rs = [r for r in rows if r["tool"] == tool and r["condition"] in TRANSIENT
              and (r["task_id"], r["seed"]) in clean_ok]
        lines.append(f"- {tool}: {pct(mean([r['outcome'] == 'grounded_success' for r in rs]))}(n={len(rs)})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--regrade", action="store_true", help="用当前判分器重新判已存的轨迹")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(summarize(load(args.path, args.regrade)))


if __name__ == "__main__":
    main()
