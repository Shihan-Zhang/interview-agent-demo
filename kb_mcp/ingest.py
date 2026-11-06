"""
把 rag/store 里已经算好的块和 bge-m3 向量导入本地 Qdrant(嵌入式模式,不需要数据库服务)。

为什么不用服务器自带的 qdrant-store 工具导入:
- 它一次只存一条,每条都要重新向量化;我们的向量已经算好(rag/index.py),直接复用,
  这样 MCP 检索和直接检索用的是同一份文档向量,差异只可能来自查询端和搜索实现;
- 它的点 ID 是随机 uuid4,重跑会产生重复数据;这里用"块文本的哈希"生成确定的 ID,重跑只覆盖。

payload 结构与 mcp-server-qdrant 一致:{"document": 文本, "metadata": {...}},服务器才能原样读出。
注意:嵌入式 Qdrant 同一个目录同一时间只能被一个进程打开 —— 导入完关闭,再启动服务器。

用法:python -m kb_mcp.ingest [--path D:\\ZSH\\cache\\qdrant] [--collection notes] [--recreate]
"""

import argparse
import hashlib
import json
import os
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from kb_mcp.ollama_provider import OllamaEmbeddingProvider
from rag.index import STORE, file_sha256

ID_NAMESPACE = uuid.UUID("6f3c2a7e-5b1d-4c8e-9a40-2d7f1e0b9c11")   # 固定命名空间:同一块文本永远得到同一个 ID


def point_id(chunk) -> str:
    """块 ID = uuid5(来源 + 文本的 SHA-256)。文本不变 ID 不变,重跑覆盖而不是新增。"""
    h = hashlib.sha256(f"{chunk['source']}\n{chunk['text']}".encode("utf-8")).hexdigest()
    return str(uuid.uuid5(ID_NAMESPACE, h))


def load_store(store=STORE):
    path = os.path.join(store, "chunks.jsonl")
    with open(path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(store, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["chunks_sha256"] != file_sha256(path):
        raise RuntimeError("chunks.jsonl 和向量不是同一版本(哈希不符),先运行 python -m rag.index")
    emb = np.load(os.path.join(store, "emb.npy"))
    if emb.shape != (len(chunks), manifest["dim"]):
        raise RuntimeError(f"向量形状 {emb.shape} 与块数 {len(chunks)} / 维度 {manifest['dim']} 不符")
    return chunks, emb, manifest


def build_points(chunks, emb, vector_name):
    pts = []
    for c, v in zip(chunks, emb):
        meta = {"chunk_id": c["id"], "source": c["source"], "doc_title": c.get("doc_title"),
                "path": c.get("path"), "start": c.get("start"), "end": c.get("end")}
        pts.append(models.PointStruct(id=point_id(c), vector={vector_name: v.astype(float).tolist()},
                                      payload={"document": c["text"], "metadata": meta}))
    ids = [p.id for p in pts]
    if len(set(ids)) != len(ids):
        raise RuntimeError("有重复的块(同来源同文本),ID 冲突")
    return pts


def ingest(client, collection, chunks, emb, provider, recreate=False):
    name, dim = provider.get_vector_name(), provider.get_vector_size()
    if emb.shape[1] != dim:
        raise RuntimeError(f"向量维度 {emb.shape[1]} 与 provider 声明的 {dim} 不符")
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config={
            name: models.VectorParams(size=dim, distance=models.Distance.COSINE)})
    client.upsert(collection, points=build_points(chunks, emb, name), wait=True)
    return client.count(collection, exact=True).count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.environ.get("QDRANT_LOCAL_PATH", r"D:\ZSH\cache\qdrant"))
    ap.add_argument("--collection", default=os.environ.get("COLLECTION_NAME", "notes"))
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()
    chunks, emb, manifest = load_store()
    provider = OllamaEmbeddingProvider(manifest["embedding_model"], manifest["dim"])
    client = QdrantClient(path=args.path)
    try:
        n = ingest(client, args.collection, chunks, emb, provider, args.recreate)
    finally:
        client.close()   # 释放目录锁,服务器才能打开
    print(f"导入 {len(chunks)} 块到 {args.path} / {args.collection}(集合现有 {n} 点),"
          f"向量 {provider.get_vector_name()} × {manifest['dim']} 维,chunks 哈希 {manifest['chunks_sha256'][:12]}")


if __name__ == "__main__":
    main()
