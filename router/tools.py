"""
路由 Agent 的三个工具:schema + 实现 + 统一的返回约定。

返回约定(所有工具一致,故障注入器也遵守它):
  成功: {"ok": True, ...数据...}
  失败: {"ok": False, "error_type": <类别>, "error": <给模型看的说明>, "retryable": bool?}
error_type 是给程序(评测、重试策略)用的,error 是给模型看的,两者分开。

设计取舍(对应 M9 第二层学过的工具设计原则):
- 工具按"数据形态"划分,而不是按"API"划分:笔记=模糊理解,岗位=确定性筛选,搜索=最新信息;
- description 写清楚"什么时候用 / 什么时候不用",这是模型路由的主要依据;
- 返回给模型的内容要精简:笔记只回块文本和出处,不回向量分数以外的内部字段;
- notes_search 返回"证据块"而不是生成好的答案:让 Agent 基于证据作答,
  答案可以追溯到块,评测也能区分"检索错"和"生成错"。
"""

from router.jobs import CITIES, LEVELS, RECRUIT_TYPES, SORT_KEYS, STATUSES, TRACKS

TOOL_SCHEMAS = [
    {
        "name": "notes_search",
        "description": "在我的个人学习笔记(大模型知识库、求职与学习计划)里检索相关段落。"
                       "适用:概念、公式、我做过的实验结论、学习计划。"
                       "不适用:岗位筛选统计(用 jobs_query)、笔记之外的最新消息(用 web_search)。"
                       "返回若干段原文和出处;回答时只根据返回的原文,并注明出处编号。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题,用笔记里可能出现的关键词"},
                "k": {"type": "integer", "description": "返回段落数,默认 5,最多 10"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "jobs_query",
        "description": "查询我的求职追踪表(结构化数据)。适用:按状态/城市/赛道/日期筛选岗位、"
                       "数一共有几个、按日期或优先级排序。不适用:笔记内容、招聘网站上的新岗位。"
                       "所有参数都可选;返回 count_total(符合条件的总数)和前 limit 行。",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": STATUSES, "description": "投递状态"},
                "city": {"type": "string", "enum": CITIES},
                "track": {"type": "string", "enum": TRACKS, "description": "赛道(岗位可属于多个赛道)"},
                "recruit_type": {"type": "string", "enum": RECRUIT_TYPES},
                "priority": {"type": "string", "enum": LEVELS},
                "keyword": {"type": "string", "description": "在公司名、岗位名、JD 里做包含匹配"},
                "applied_from": {"type": "string", "description": "投递日期下限,YYYY-MM-DD(含)"},
                "applied_to": {"type": "string", "description": "投递日期上限,YYYY-MM-DD(含)"},
                "next_before": {"type": "string", "description": "下次节点(笔试/面试/截止)不晚于此日期,YYYY-MM-DD"},
                "sort_by": {"type": "string", "enum": SORT_KEYS},
                "limit": {"type": "integer", "description": "最多返回几行,默认 10,最多 50"},
            },
            "required": [],
        },
    },
    {
        "name": "web_search",
        "description": "联网搜索最新信息:新发布的岗位、行业新闻、公司动态。"
                       "适用:笔记和岗位表里没有、或需要最新消息的问题。返回标题、链接、摘要和发布日期。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "默认 5"},
            },
            "required": ["query"],
        },
    },
]


def to_openai_schemas(schemas=TOOL_SCHEMAS):
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                              "parameters": s["input_schema"]}} for s in schemas]


def build_tools(retriever=None, jobs_table=None, search_backend=None, notes_mode="hybrid"):
    """依赖注入:后端从外面传进来。单元测试传假后端,评测传固定数据,日常使用传真实后端。"""

    def notes_search(query: str, k: int = 5):
        if retriever is None:
            return {"ok": False, "error_type": "unavailable", "error": "笔记索引未加载", "retryable": False}
        k = max(1, min(int(k), 10))
        chunks = retriever.search(query, k=k, mode=notes_mode)
        return {"ok": True, "results": [
            {"ref": f"[{i}]", "source": " > ".join(c.get("path", [])) or c.get("source", ""),
             "text": c["text"]} for i, c in enumerate(chunks, 1)]}

    def jobs_query(**filters):
        if jobs_table is None:
            return {"ok": False, "error_type": "unavailable", "error": "岗位表未加载", "retryable": False}
        unknown = set(filters) - set(TOOL_SCHEMAS[1]["input_schema"]["properties"])
        if unknown:
            return {"ok": False, "error_type": "invalid_arguments",
                    "error": f"未知参数 {sorted(unknown)},可用参数: "
                             f"{list(TOOL_SCHEMAS[1]['input_schema']['properties'])}"}
        return jobs_table.query(**filters)

    def web_search(query: str, max_results: int = 5):
        if search_backend is None:
            return {"ok": False, "error_type": "unavailable", "error": "搜索后端未配置", "retryable": False}
        return {"ok": True, "results": search_backend.search(query, max(1, min(int(max_results), 10)))}

    return {"notes_search": notes_search, "jobs_query": jobs_query, "web_search": web_search}


SYSTEM_PROMPT_BASE = """你是我的求职与学习助理。今天是 {today}。
你有三个工具:notes_search(我的学习笔记)、jobs_query(我的求职追踪表)、web_search(联网搜索)。
事实性问题先用工具查,只根据工具返回的内容回答;引用笔记时注明出处编号,如 [1]。"""

# 带"出错怎么办"规则的版本。两版的差别就是故障评测里的一个因素:
# 恢复能力有多少来自模型本身、多少来自提示词。
SYSTEM_PROMPT = """你是我的求职与学习助理。今天是 {today}。
你有三个工具:notes_search(我的学习笔记)、jobs_query(我的求职追踪表)、web_search(联网搜索)。
规则:
1. 事实性问题先用工具查,不要凭记忆回答我的笔记、岗位表或最新消息。
2. 只根据工具返回的内容回答;引用笔记时注明出处编号,如 [1]。
3. 工具报错时:如果是暂时性错误(超时、限流、服务错误),可以重试一次;参数错误就按提示改参数;
   一个工具一直不可用,看看其他工具能否回答;都不行就直接告诉我"暂时查不到"以及原因,不要编造。
4. 返回内容看起来残缺或为空时,不要当成"没有这条数据"下结论,可以换个查询再试一次。"""
