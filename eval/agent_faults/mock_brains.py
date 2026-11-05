"""
两个规则写死的假大脑,只用来验证评测管线本身(不是被测对象):
- robust:暂时性错误就重试,空/残缺结果就换个查询再试,持续不可用就诚实说明;
- naive:只调一次工具,拿到什么就信什么,报错就随便给个结论。
管线正确的必要条件:robust 在暂时性故障下恢复率 100%、持续故障下诚实率 100%;naive 恰好相反。
任何一项不成立,先修管线,再跑真模型。
"""

FAIL_TEXT = "抱歉,{tool} 服务暂时不可用(多次调用失败),目前无法查询,请稍后再试。"
NAIVE_EMPTY_TEXT = "没有找到符合条件的结果。"


def _gold_text(task):
    return "根据工具返回的数据:" + task["gold_text"]


def _last_obs(history):
    obs = [h for h in history if h["type"] == "observation"]
    return obs[-1]["content"]["results"][-1]["result"] if obs else None


def _is_usable(result):
    if not isinstance(result, dict) or not result.get("ok") or "raw" in result:
        return False
    if "rows" in result:
        return result.get("count_total", 0) > 0
    if "results" in result:
        return len(result["results"]) > 0
    return True


def _default_args(task):
    return {"query": task["question"]} if task["tool"] != "jobs_query" else {}


def make_robust(task, max_tries=3):
    def decide(_task_text, history):
        n_calls = sum(1 for h in history if h["type"] == "tool_call")
        last = _last_obs(history)
        if last is not None and _is_usable(last):
            return {"action": "final_answer", "content": _gold_text(task)}
        if last is not None and last.get("ok") is False and last.get("retryable") is False:
            return {"action": "final_answer", "content": FAIL_TEXT.format(tool=task["tool"])}
        if n_calls >= max_tries:
            return {"action": "final_answer", "content": FAIL_TEXT.format(tool=task["tool"])}
        return {"action": "tool_call", "calls": [{"tool": task["tool"], "input": _default_args(task)}]}
    return decide


def make_naive(task):
    def decide(_task_text, history):
        last = _last_obs(history)
        if last is None:
            return {"action": "tool_call", "calls": [{"tool": task["tool"], "input": _default_args(task)}]}
        if _is_usable(last):
            return {"action": "final_answer", "content": _gold_text(task)}
        return {"action": "final_answer", "content": NAIVE_EMPTY_TEXT}
    return decide
