"""
等价性评测:通过 MCP(kb_mcp.server,stdio 子进程)检索 vs 直接调用 rag/retrieve.py 的 dense 检索。

两边用同一份文档向量(kb_mcp.ingest 直接导入 rag/store/emb.npy)、同一个查询向量函数(rag.index.embed),
嵌入式 Qdrant 是精确检索,所以理论上逐题排名应完全一致。报告:
- 两边的 Recall@1/3/5、MRR@10(与 eval_retrieval.py 同一标注、同一命中规则);
- 逐题 top-10 列表完全相同的题数,以及第一个不一致的位置;
- 命中@5 的配对 McNemar。
有任何不一致,先查管线(块是否同一版本、向量名、查询向量是否同参数),再下结论。

运行(先运行 python -m kb_mcp.ingest):
  python eval/rag/eval_mcp_retrieval.py [--out eval/results/rag_mcp_equivalence_report.md]
"""

import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval", "fc"))

from eval_retrieval import QUESTIONS, check_labels, first_hit_rank  # noqa: E402
from summarize_fc import mcnemar_exact  # noqa: E402

from kb_mcp.results import chunk_ids, find_text  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

DEPTH = 10


async def mcp_search_all(queries, env):
    from fastmcp import Client
    from fastmcp.client.transports import PythonStdioTransport
    transport = PythonStdioTransport(script_path=os.path.join(ROOT, "kb_mcp", "server.py"), env=env, cwd=ROOT)
    out, lat = {}, []
    async with Client(transport) as c:
        tools = sorted(t.name for t in await c.list_tools())
        for qid, text in queries:
            t0 = time.perf_counter()
            res = await c.call_tool("qdrant-find", {"query": text})
            lat.append(time.perf_counter() - t0)
            out[qid] = chunk_ids(find_text(res))
    return out, tools, lat


def metrics(ranks, k):
    n = len(ranks)
    rec = lambda kk: sum(1 for x in ranks if x and x <= kk) / max(n, 1)  # noqa: E731
    return rec(1), rec(3), rec(k), sum(1 / x for x in ranks if x) / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "notes"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results", "rag_mcp_equivalence_report.md"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    corpus_text = ""
    for dp, _, fs in os.walk(args.corpus):
        for fn in fs:
            if fn.endswith(".md"):
                with open(os.path.join(dp, fn), encoding="utf-8") as f:
                    corpus_text += f.read() + "\n"
    r = Retriever.load(cpu=True)
    usable, problems = check_labels(questions, r, corpus_text)
    by_id = {c["id"]: c for c in r.chunks}

    direct = {q["id"]: [c["id"] for c in r.search(q["question"], k=DEPTH, mode="dense")] for q in usable}
    env = dict(os.environ, QDRANT_READ_ONLY="true", QDRANT_SEARCH_LIMIT=str(DEPTH), PYTHONPATH=ROOT)
    env.setdefault("QDRANT_LOCAL_PATH", r"D:\ZSH\cache\qdrant")
    env.setdefault("COLLECTION_NAME", "notes")
    via_mcp, tools, lat = asyncio.run(mcp_search_all([(q["id"], q["question"]) for q in usable], env))

    def ranks(lists):
        return {q["id"]: first_hit_rank([by_id[i] for i in lists[q["id"]] if i in by_id], q["evidence"]) for q in usable}
    rd, rm = ranks(direct), ranks(via_mcp)
    same = [q["id"] for q in usable if direct[q["id"]] == via_mcp[q["id"]]]
    diffs = []
    for q in usable:
        a, b = direct[q["id"]], via_mcp[q["id"]]
        if a != b:
            pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            diffs.append((q["id"], pos + 1, rd[q["id"]], rm[q["id"]]))
    hd = [bool(rd[q["id"]] and rd[q["id"]] <= args.k) for q in usable]
    hm = [bool(rm[q["id"]] and rm[q["id"]] <= args.k) for q in usable]
    oa = sum(x and not y for x, y in zip(hd, hm))
    ob = sum(y and not x for x, y in zip(hd, hm))

    L = ["# 知识库 MCP 等价性评测\n",
         f"索引:{r.manifest.get('embedding_model')},{len(r.chunks)} 块,chunks 哈希 `{r.manifest.get('chunks_sha256', '')[:12]}`;"
         f"可评题 {len(usable)}(标注有问题 {len(problems)})。MCP 服务器暴露的工具:{', '.join(tools)}。\n",
         f"| 路径 | Recall@1 | Recall@3 | Recall@{args.k} | MRR@{DEPTH} |", "|---|---|---|---|---|"]
    for name, rk in (("直接调用 retrieve.py(dense)", rd), ("通过 MCP(qdrant-find)", rm)):
        a, b, c, d = metrics([rk[q["id"]] for q in usable], args.k)
        L.append(f"| {name} | {a:.0%} | {b:.0%} | {c:.0%} | {d:.3f} |")
    L += ["", f"- 逐题 top-{DEPTH} 列表完全相同:**{len(same)}/{len(usable)}**",
          f"- 命中@{args.k} 配对:只有直接命中 {oa},只有 MCP 命中 {ob},精确 McNemar p = {mcnemar_exact(oa, ob):.3f}",
          f"- MCP 单次检索耗时(含查询向量化,CPU):中位数 {sorted(lat)[len(lat) // 2]:.2f} s,最长 {max(lat):.2f} s"]
    if diffs:
        L += ["", "## 不一致的题", "", "| 题 | 第一个不同的位置 | 直接命中排名 | MCP 命中排名 |", "|---|---|---|---|"]
        L += [f"| {i} | {p} | {a} | {b} |" for i, p, a, b in diffs]
    text = "\n".join(L) + "\n"
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
