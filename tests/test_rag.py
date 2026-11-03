"""RAG 切分与检索的基本性质。"""

import os

import numpy as np

from rag.ingest import chunk_document, parse_sections
from rag.retrieve import Retriever, rrf, top_k
from rag.text import split_sentences, tokenize

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sample_notes.md")
with open(FIXTURE, encoding="utf-8") as f:
    TEXT = f.read()


def test_headings_inside_code_fence_are_not_sections():
    title, sections = parse_sections(TEXT)
    assert title == "测试知识库"
    paths = [p for p, _, _ in sections]
    assert ["M5 评测统计", "精确 McNemar 检验"] in paths
    assert not any("这一行以井号开头" in " ".join(p) for p in paths)


def test_chunk_offsets_point_back_to_source():
    # 评测按原文位置判断命中,所以每个块的每个片段都必须能在 [start, end) 里找到
    for c in chunk_document(TEXT, "sample.md", max_tokens=60):
        span = TEXT[c["start"]:c["end"]]
        for piece in c["text"].split("\n"):
            assert piece.strip() in span, (piece, span)


def test_code_block_kept_whole_and_path_prefixed():
    chunks = chunk_document(TEXT, "sample.md", max_tokens=400)
    code = [c for c in chunks if "print(" in c["text"]][0]
    block = '```python\n# 这一行以井号开头，但它在代码块里，不是标题\nprint("not a heading")\n```'
    assert block in code["text"]                     # 代码块完整,没有被按句子拆开
    assert code["embed_text"].startswith("[M5 评测统计 > 精确 McNemar 检验]")
    assert not any(c["embed_text"].startswith("[测试知识库") for c in chunks)   # 不带一级标题


def test_small_budget_splits_with_sentence_overlap():
    chunks = [c for c in chunk_document(TEXT, "sample.md", max_tokens=40) if c["path"][-1:] == ["精确 McNemar 检验"]]
    assert len(chunks) >= 2
    assert chunks[1]["start"] < chunks[0]["end"]          # 相邻块有重叠


def test_split_sentences_on_chinese_punctuation():
    s = "第一句。第二句！\n第三句"
    assert [s[a:b] for a, b in split_sentences(s)] == ["第一句。", "第二句！", "第三句"]


def test_userdict_keeps_model_names_whole():
    toks = tokenize("Qwen3-4B-Instruct-2507 的推荐温度")
    assert "qwen3-4b-instruct-2507" in toks
    assert "的" not in toks


def test_rrf_and_top_k():
    assert rrf([[0, 1, 2], [2, 0, 3]])[:2] == [0, 2]
    assert top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2) == [1, 3]


def test_bm25_finds_exact_model_name():
    chunks = chunk_document(TEXT, "sample.md")
    r = Retriever(chunks)                          # 不带向量,只测 BM25
    best = r.search("Qwen3-4B-Instruct-2507 推荐温度", k=1, mode="bm25")[0]
    assert "0.7" in best["text"]


def test_bm25_never_returns_zero_score_chunks():
    chunks = chunk_document(TEXT, "sample.md")
    r = Retriever(chunks)
    assert r.search("完全不相关的词汇组合", k=5, mode="bm25") == []


def test_tiny_sections_are_dropped():
    text = "# 标题\n\nNov 2, 2026 · @Someone\n\n## 正文\n\n" + "这是一段足够长的正文内容，用来确保它会成为一块。" * 3
    chunks = chunk_document(text, "t.md")
    assert len(chunks) == 1 and chunks[0]["path"] == ["正文"]
