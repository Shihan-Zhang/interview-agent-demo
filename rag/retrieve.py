"""
检索:向量 / BM25 / RRF 融合,每个组件都可以单独开关,方便做消融实验。重排(cross-encoder)下一步加。

    r = Retriever.load()
    r.search("McNemar 检验怎么算", k=5, mode="hybrid")   # mode: dense | bm25 | hybrid

运行(调试用):python -m rag.retrieve "你的问题" [--mode dense|bm25|hybrid] [--cpu]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from rag.index import STORE, embed, file_sha256
    from rag.text import tokenize
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag.index import STORE, embed, file_sha256
    from rag.text import tokenize


def rrf(rank_lists, k=60):
    """倒数排名融合:每个检索器给出一个排好序的块下标列表;没出现在某个列表里的块,那一项记 0。"""
    score = defaultdict(float)
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, start=1):
            score[idx] += 1.0 / (k + r)
    return sorted(score, key=lambda i: (-score[i], i))


def top_k(scores, k):
    """取分数最高的 k 个下标:argpartition O(N) 先选,再只对这 k 个排序。"""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")].tolist()


class Retriever:
    def __init__(self, chunks, emb=None, manifest=None, cpu=True):
        self.chunks, self.emb, self.manifest, self.cpu = chunks, emb, manifest or {}, cpu
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])

    @classmethod
    def load(cls, store=STORE, cpu=True):
        path = os.path.join(store, "chunks.jsonl")
        with open(path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(store, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["chunks_sha256"] != file_sha256(path):
            raise RuntimeError("chunks.jsonl 和向量不是同一版本(哈希不符),请重新运行 python -m rag.index")
        emb = np.load(os.path.join(store, "emb.npy"))
        assert emb.shape == (len(chunks), manifest["dim"]), (emb.shape, len(chunks))
        return cls(chunks, emb, manifest, cpu)

    def dense_scores(self, query):
        q = embed([query], self.manifest.get("embedding_model", "bge-m3"), cpu=self.cpu)[0]   # [dim]
        return self.emb @ q                                                                      # [N]

    def bm25_scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)               # [N]

    def search(self, query, k=5, mode="hybrid", pool=30):
        if mode == "dense":
            order = top_k(self.dense_scores(query), k)
        elif mode == "bm25":
            order = top_k(self.bm25_scores(query), k)
        elif mode == "hybrid":
            order = rrf([top_k(self.dense_scores(query), pool), top_k(self.bm25_scores(query), pool)])[:k]
        else:
            raise ValueError(mode)
        return [self.chunks[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--gpu", action="store_true", help="查询向量在 GPU 上算(默认 CPU,给生成模型留显存)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    r = Retriever.load(cpu=not args.gpu)
    for rank, c in enumerate(r.search(args.query, args.k, args.mode), 1):
        print(f"[{rank}] {c['source']} | {' > '.join(c['path'])} | {c['start']}-{c['end']}")
        print("    " + c["text"][:160].replace("\n", " ") + ("…" if len(c["text"]) > 160 else ""))


if __name__ == "__main__":
    main()
