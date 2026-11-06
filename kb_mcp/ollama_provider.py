"""
mcp-server-qdrant 的 Ollama 向量化实现。

官方服务器把向量化抽象成 EmbeddingProvider(4 个方法),但只自带 FastEmbed 一种实现,
默认是英文小模型 all-MiniLM-L6-v2。这里补一个 Ollama 实现,接上我们的中文 bge-m3。

查询向量直接调用 rag.index.embed:和 rag/retrieve.py 用同一个函数、同一组参数(CPU、逐行归一化),
保证"通过 MCP 检索"和"直接调用检索"拿到的是同一个查询向量,等价性评测才有意义。
"""

import asyncio

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

from rag.index import embed


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "bge-m3", dim: int = 1024, cpu: bool = True):
        self.model, self.dim, self.cpu = model, dim, cpu

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        # embed 是同步的 HTTP 调用,放到线程池里,避免阻塞服务器的事件循环(官方 FastEmbed 实现也是这么做的)
        vecs = await asyncio.to_thread(embed, documents, self.model, self.cpu)
        return vecs.tolist()

    async def embed_query(self, query: str) -> list[float]:
        vecs = await asyncio.to_thread(embed, [query], self.model, self.cpu)
        return vecs[0].tolist()

    def get_vector_name(self) -> str:
        # 集合里的"命名向量"名;导入脚本和服务器必须一致,否则检索时找不到向量
        return f"ollama-{self.model.replace(':', '-').replace('/', '-').lower()}"

    def get_vector_size(self) -> int:
        return self.dim
