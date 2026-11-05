"""
路由工具、故障注入器、故障判分器的测试——测的是"评测工具本身"对不对。
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_faults"))

from agent import execute_tool, run_agent  # noqa: E402
from faults import FaultInjector  # noqa: E402
from grader import answer_matches, classify  # noqa: E402
from mock_brains import make_naive, make_robust  # noqa: E402
from router.jobs import JobsTable, load_jobs  # noqa: E402
from router.search import ReplayBackend  # noqa: E402
from router.tools import build_tools  # noqa: E402

FIX = os.path.join(ROOT, "eval", "agent_faults", "fixtures")
TASKS = [json.loads(l) for l in open(os.path.join(ROOT, "eval", "agent_faults", "tasks_v0.jsonl"), encoding="utf-8")]


def jobs():
    return JobsTable(load_jobs(os.path.join(FIX, "jobs_fixture.json")))


class FakeRetriever:
    def search(self, query, k=5, mode="hybrid"):
        return [{"path": ["M5", "McNemar"], "text": "b = 6、c = 1 时 p = 0.125"}][:k]


def tools(**kw):
    search = ReplayBackend.load(os.path.join(FIX, "search_replay.jsonl"))
    search.task_id = "web-03"
    return build_tools(retriever=FakeRetriever(), jobs_table=jobs(), search_backend=search, **kw)


# ---------- 岗位表 ----------

def test_jobs_invalid_enum_returns_actionable_error():
    r = jobs().query(city="成都")
    assert r["ok"] is False and r["error_type"] == "invalid_arguments"
    assert "上海" in r["error"]           # 报错里给出可选值,模型才能据此改参数


def test_jobs_count_total_is_independent_of_limit():
    r = jobs().query(city="上海", status="已投递", limit=1)
    assert r["count_total"] == 5 and len(r["rows"]) == 1


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


# ---------- 故障注入器 ----------

def test_transient_fault_only_hits_first_call():
    inj = FaultInjector(tools(), "notes_search", "timeout")
    t = inj.wrap()
    assert t["notes_search"](query="x")["error_type"] == "timeout"
    assert t["notes_search"](query="x")["ok"] is True
    assert inj.genuine_success()


def test_unavailable_persists_and_never_counts_as_genuine():
    inj = FaultInjector(tools(), "jobs_query", "unavailable")
    t = inj.wrap()
    assert all(t["jobs_query"]()["error_type"] == "unavailable" for _ in range(3))
    assert not inj.genuine_success()


def test_empty_fault_keeps_shape():
    t = FaultInjector(tools(), "jobs_query", "empty").wrap()
    r = t["jobs_query"](city="上海")
    assert r["ok"] is True and r["rows"] == [] and r["count_total"] == 0


def test_malformed_payload_is_not_valid_json():
    t = FaultInjector(tools(), "web_search", "malformed").wrap()
    r = t["web_search"](query="LZ-Bench")
    try:
        json.loads(r["raw"])
        assert False, "截断后的载荷不应能被解析"
    except json.JSONDecodeError:
        pass


def test_other_tools_are_not_injected():
    inj = FaultInjector(tools(), "web_search", "unavailable")
    assert inj.wrap()["jobs_query"]()["ok"] is True


# ---------- 判分器自洽 ----------

def test_every_gold_text_passes_its_own_grader():
    bad = [t["id"] for t in TASKS if not answer_matches(t, t["gold_text"])]
    assert bad == []


def _final(text):
    return {"type": "final_answer", "content": {"content": text}}


def test_honest_failure_vs_false_negative():
    task = next(t for t in TASKS if t["id"] == "jobs-01")
    assert classify(task, _final("岗位表查询服务暂时不可用,目前无法查询。"), False) == "honest_failure"
    # "没有找到"断言数据不存在,而不是归因于工具:空结果注入下这是错误结论
    assert classify(task, _final("没有找到符合条件的岗位,共 0 个。"), False) == "wrong_answer"


def test_listing_every_company_does_not_pass_set_questions():
    task = next(t for t in TASKS if t["id"] == "jobs-05")
    all_cos = "、".join(r["company"] for r in jobs().rows)
    assert classify(task, _final(all_cos), True) == "wrong_answer"


def test_correct_answer_without_genuine_data_is_unsupported():
    task = next(t for t in TASKS if t["id"] == "web-03")
    assert classify(task, _final("一共 820 道任务"), False) == "unsupported_correct"


def test_max_steps_is_its_own_outcome():
    task = TASKS[0]
    step = {"type": "final_answer", "content": {"content": "...", "max_steps_reached": True}}
    assert classify(task, step, False) == "max_steps"


# ---------- 端到端:假大脑验证管线 ----------

def _run(brain_factory, task, cond):
    inj = FaultInjector(tools(), task["tool"], cond)
    traj = run_agent(task["question"], decide_fn=brain_factory(task), verbose=False,
                     tool_functions=inj.wrap(), max_steps=8)
    return classify(task, traj.steps[-1], inj.genuine_success())


def test_pipeline_separates_robust_from_naive_brain():
    task = next(t for t in TASKS if t["id"] == "notes-02")
    for cond in ["timeout", "rate_limited", "malformed", "empty"]:
        assert _run(make_robust, task, cond) == "grounded_success", cond
        assert _run(make_naive, task, cond) == "wrong_answer", cond
    assert _run(make_robust, task, "unavailable") == "honest_failure"
    assert _run(make_naive, task, "unavailable") == "wrong_answer"
