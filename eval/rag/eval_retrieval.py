"""
检索评测:Recall@k、MRR,三种模式配对比较。

标注方式:每道题标一句"证据原文"(evidence),某块的文字里包含这句证据,就算命中。
不标块编号:换了切分参数,块编号全变,标注就失效;证据原文和切分方案无关(见 M8 工程第 7 点)。
比较时先做归一化(去空白、反斜杠、Markdown 的 * 和 `),因为导出的 Markdown 会把 d_k 转义成 d\\_k。

先做标注检查(和 BFCL 的自洽检查同一个思路):
- 证据必须能在语料里找到,否则是标注写错了;
- 证据必须完整落在至少一块里,否则是被切分切断了(换短一点的证据,或者记下来作为切分的问题)。
检查不通过的题不参与打分,并在报告里列出来。

"查不到"的题(evidence 为空)不算 Recall,只记录它们的最高检索分数,
和能回答的题对比,看"按分数阈值拒答"是否可行。

运行:python eval/rag/eval_retrieval.py [--modes dense,bm25,hybrid] [-k 5]
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "fc"))
from rag.retrieve import Retriever, top_k  # noqa: E402
from summarize_fc import mcnemar_exact  # noqa: E402

QUESTIONS = os.path.join(HERE, "questions_v0.jsonl")
REPORT = os.path.join(ROOT, "eval", "results", "rag_retrieval_report.md")


def norm(s):
    return re.sub(r"[\s\\*`]", "", s).lower()


def first_hit_rank(chunks, evidence):
    e = norm(evidence)
    for r, c in enumerate(chunks, start=1):
        if e in norm(c["text"]):
            return r
    return None


def check_labels(questions, retriever, corpus_text):
    """返回 (可用的题, 问题列表)。"""
    ok, problems = [], []
    corpus_n = norm(corpus_text)
    for q in questions:
        if not q["evidence"]:
            continue
        e = norm(q["evidence"])
        if e not in corpus_n:
            problems.append((q["id"], "证据在语料里找不到(标注写错或语料没更新)"))
        elif not any(e in norm(c["text"]) for c in retriever.chunks):
            problems.append((q["id"], "证据被切分切断,没有完整落在任何一块里"))
        else:
            ok.append(q)
    return ok, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="dense,bm25,hybrid")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "notes"))
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    corpus_text = ""
    for dp, _, fs in os.walk(args.corpus):
        for fn in fs:
            if fn.endswith(".md"):
                with open(os.path.join(dp, fn), encoding="utf-8") as f:
                    corpus_text += f.read() + "\n"
    modes = args.modes.split(",")
    r = Retriever.load(cpu=not args.gpu) if "dense" in modes or "hybrid" in modes else None
    if r is None:
        from rag.ingest import STORE
        with open(os.path.join(STORE, "chunks.jsonl"), encoding="utf-8") as f:
            r = Retriever([json.loads(line) for line in f if line.strip()])

    usable, problems = check_labels(questions, r, corpus_text)
    unanswerable = [q for q in questions if not q["evidence"]]
    depth = 10
    ranks = {m: {} for m in modes}
    for m in modes:
        for q in usable:
            ranks[m][q["id"]] = first_hit_rank(r.search(q["question"], k=depth, mode=m), q["evidence"])

    L = ["# 检索评测(v0)\n",
         f"索引:{r.manifest.get('embedding_model', '-')},{len(r.chunks)} 块,chunks 哈希 "
         f"`{r.manifest.get('chunks_sha256', '-')[:12]}`。题目 {len(questions)} 道:"
         f"可评 {len(usable)},查不到 {len(unanswerable)},标注有问题 {len(problems)}。\n"]
    if problems:
        L += ["## 标注检查未通过(不参与打分)\n", "| 题 | 问题 |", "|---|---|"]
        L += [f"| {i} | {p} |" for i, p in problems]
        L.append("")

    L += ["## 总体\n", f"| 模式 | Recall@1 | Recall@3 | Recall@{args.k} | MRR@{depth} |", "|---|---|---|---|---|"]
    for m in modes:
        rk = [ranks[m][q["id"]] for q in usable]
        n = len(rk)
        rec = lambda k: sum(1 for x in rk if x and x <= k) / max(n, 1)  # noqa: E731
        mrr = sum(1 / x for x in rk if x) / max(n, 1)
        L.append(f"| {m} | {rec(1):.0%} | {rec(3):.0%} | {rec(args.k):.0%} | {mrr:.3f} |")

    types = sorted({q["type"] for q in usable})
    L += [f"\n## 分题型 Recall@{args.k}\n", "| 题型 | 题数 | " + " | ".join(modes) + " |",
          "|---|---|" + "---|" * len(modes)]
    for t in types:
        qs = [q for q in usable if q["type"] == t]
        cells = [f"{sum(1 for q in qs if ranks[m][q['id']] and ranks[m][q['id']] <= args.k)}/{len(qs)}" for m in modes]
        L.append(f"| {t} | {len(qs)} | " + " | ".join(cells) + " |")

    L += [f"\n## 配对比较(命中@{args.k},精确 McNemar)\n", "| 比较 | 只有 A 命中 | 只有 B 命中 | p 值 |", "|---|---|---|---|"]
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            ha = [bool(ranks[a][q["id"]] and ranks[a][q["id"]] <= args.k) for q in usable]
            hb = [bool(ranks[b][q["id"]] and ranks[b][q["id"]] <= args.k) for q in usable]
            oa = sum(x and not y for x, y in zip(ha, hb))
            ob = sum(y and not x for x, y in zip(ha, hb))
            L.append(f"| {a} vs {b} | {oa} | {ob} | {mcnemar_exact(oa, ob):.3f} |")

    if "dense" in modes and r.emb is not None and unanswerable:
        top_ans = [float(np.max(r.dense_scores(q["question"]))) for q in usable]
        top_none = [float(np.max(r.dense_scores(q["question"]))) for q in unanswerable]
        L += ["\n## 能否靠相似度阈值拒答(dense 最高余弦)\n", "| 题 | 题数 | 最小 | 中位数 | 最大 |", "|---|---|---|---|---|",
              f"| 能回答 | {len(top_ans)} | {min(top_ans):.3f} | {np.median(top_ans):.3f} | {max(top_ans):.3f} |",
              f"| 查不到 | {len(top_none)} | {min(top_none):.3f} | {np.median(top_none):.3f} | {max(top_none):.3f} |",
              "\n两组区间重叠越多,单靠阈值拒答越不可行,拒答要交给生成阶段。"]

    L += [f"\n## 逐题排名(前 {depth} 名内首个命中的名次,— 表示没命中)\n",
          "| 题 | 题型 | 问题 | " + " | ".join(modes) + " |", "|---|---|---|" + "---|" * len(modes)]
    for q in usable:
        L.append(f"| {q['id']} | {q['type']} | {q['question']} | " +
                 " | ".join(str(ranks[m][q["id"]] or "—") for m in modes) + " |")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n报告:{REPORT}")


if __name__ == "__main__":
    main()
