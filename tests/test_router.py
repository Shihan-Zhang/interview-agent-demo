"""
路由工具与 Agent 循环兜底的单元测试。
这里的几行岗位数据只是单元测试用的输入(测代码逻辑),不是评测集;评测一律用公开测试集。
"""

from agent import execute_tool  # noqa: E402
from router.jobs import JobsTable  # noqa: E402
from router.search import SearchBackend  # noqa: E402
from router.tools import build_tools  # noqa: E402


def _row(company, city, status, applied=None, nxt=None):
    return {"company": company, "title": "岗位", "city": city, "status": status, "tracks": [],
            "recruit_type": "校招", "priority": "高", "applied_date": applied, "next_date": nxt}


ROWS = [_row("A", "上海", "已投递", "2026-10-08"), _row("B", "上海", "已投递", "2026-10-20", "2026-11-12"),
        _row("C", "北京", "一面", "2026-10-10", "2026-11-09"), _row("D", "上海", "待投递")]


def jobs():
    return JobsTable(ROWS)


class StaticSearch(SearchBackend):
    def search(self, query, max_results=5):
        return [{"title": "t", "url": "u", "snippet": "s"}][:max_results]


class FakeRetriever:
    def search(self, query, k=5, mode="hybrid"):
        return [{"path": ["M5", "McNemar"], "text": "b = 6、c = 1 时 p = 0.125"}][:k]


def tools(**kw):
    return build_tools(retriever=FakeRetriever(), jobs_table=jobs(), search_backend=StaticSearch(), **kw)


# ---------- 岗位表 ----------

def test_jobs_invalid_enum_returns_actionable_error():
    r = jobs().query(city="成都")
    assert r["ok"] is False and r["error_type"] == "invalid_arguments"
    assert "上海" in r["error"]           # 报错里给出可选值,模型才能据此改参数


def test_jobs_count_total_is_independent_of_limit():
    r = jobs().query(city="上海", status="已投递", limit=1)
    assert r["count_total"] == 2 and len(r["rows"]) == 1


def test_jobs_sort_puts_missing_dates_last_without_crashing():
    rows = jobs().query(sort_by="next_date", limit=50)["rows"]
    dated = [r["next_date"] for r in rows if r["next_date"]]
    assert dated == sorted(dated) and rows[-1]["next_date"] is None


def test_jobs_tool_rejects_unknown_parameter():
    r = tools()["jobs_query"](location="上海")
    assert r["ok"] is False and "city" in r["error"]


def test_bad_date_format_is_reported():
    assert jobs().query(applied_from="10月1日")["error_type"] == "invalid_arguments"


# ---------- Agent 循环兜底 ----------

def test_wrong_argument_name_becomes_observation_not_crash():
    r = execute_tool(tools(), "notes_search", {"q": "McNemar"})
    assert r["ok"] is False and r["error_type"] == "invalid_arguments"


def test_tool_exception_becomes_observation():
    def boom(**_):
        raise RuntimeError("disk")
    assert execute_tool({"boom": boom}, "boom", {})["error_type"] == "tool_exception"
