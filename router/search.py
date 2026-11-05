"""
web_search 的后端。工具函数只依赖 SearchBackend 接口,真实后端(博查 / Tavily / DuckDuckGo)以后再接。

评测一律用 ReplayBackend:回放录好的固定结果。理由——
1. 可复现:真实搜索结果每天都变,今天的失败明天可能复现不了;
2. 隔离变量:故障注入评测测的是"出错后能不能恢复",不是搜索引擎好不好;
3. 不花钱、不受网络影响(国内访问部分搜索 API 需要代理)。

回放的匹配方式:按 task_id 取该题录好的结果,不按查询字符串精确匹配。
因为模型每次改写的查询都不一样,精确匹配会把"查询写法不同"混成"搜不到",
这是评测管线引入的假失败。代价:测不出模型查询写得好不好(已在报告里注明)。
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
