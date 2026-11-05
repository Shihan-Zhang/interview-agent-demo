"""
web_search 的后端。工具函数只依赖 SearchBackend 接口,真实后端(博查 / Tavily / DuckDuckGo)以后再接。

ReplayBackend:回放录下来的真实搜索结果(按 task_id 取),用于需要可复现的场景。
"""

import json


class SearchBackend:
    def search(self, query: str, max_results: int = 5) -> list:
        raise NotImplementedError


class ReplayBackend(SearchBackend):
    def __init__(self, records: dict):
        self.records = records          # task_id -> [ {title, url, snippet, published} ]
        self.task_id = None             # 评测运行器在每道题开始前设置

    @classmethod
    def load(cls, path: str):
        records = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["task_id"]] = r["results"]
        return cls(records)

    def search(self, query: str, max_results: int = 5) -> list:
        return list(self.records.get(self.task_id, []))[:max_results]
