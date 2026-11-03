"""
判分器第二版 + 操纵检查。直接作用在已保存的轨迹上,改判分规则不需要重新跑模型。

第一版的问题(来自对 15 次失败轨迹的逐条阅读):
1. 只看第一次工具调用的结果。模型第一次用英文术语查询失败、自己改用中文重查成功并答对,
   仍被判失败(5 次误判)。
2. "如实报告失败"靠关键词表,"并没有收录"这种说法不在表里就被判失败(1 次误判)。
3. 要求必须调用工具。模型不调用工具、直接说明查不到且没有编造,结果上是对的,也被判失败(3 次)。
   按"判结果,不判路径"的原则,这类题改为只看结果。

第二版的规则:
- numeric:任一次计算器调用成功且结果正确;
- term:任一次概念查询成功且查到了期望的词条;
- tool_error_handled:最终回答明确表示无法完成/查不到(不再要求必须调用工具)。
另外记录几个诊断字段,用于检查判分器是否把错的判成对(抽查成功样本时用)。

操纵检查:关思考的配置,不能有单独记录的思考内容,回答里也不能有非空的 <think> 段落。
"""

import re

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)

FAILURE_SIGNALS = [
    "错误", "失败", "无法", "不存在", "未找到", "没有找到", "找不到", "未收录", "没有收录",
    "不在", "超出", "没有关于", "未能", "抱歉", "不能计算", "没有定义", "无定义", "未定义",
    "division by zero", "undefined", "error", "not found",
]


def final_text(traj: dict) -> str:
    texts = [s["content"].get("content", "") for s in traj["steps"] if s["type"] == "final_answer"]
    return texts[-1] if texts else ""


def strip_think(text: str) -> str:
    """去掉回答里的 <think>...</think> 段落(包括空段落),只留下真正给用户看的部分。
    只有结束标签、没有开始标签时(思考泄漏进正文的常见形式),取最后一个结束标签之后的内容。"""
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def manipulation_check(row: dict) -> bool:
    """处理条件是否真的生效。关思考:不能有思考内容;开思考:应该有思考内容。"""
    if row.get("status") != "ok":
        return False
    text = final_text(row["trajectory"])
    leaked = any(m.strip() for m in THINK_BLOCK.findall(text)) or ("</think>" in text and "<think>" not in text)
    thinking = (row.get("usage") or {}).get("thinking_chars", 0) or 0
    if row["think"]:
        return thinking > 0 and not leaked
    return thinking == 0 and not leaked


def _calls_and_results(traj: dict):
    calls = {s["content"]["id"]: s["content"] for s in traj["steps"] if s["type"] == "tool_call"}
    pairs = []
    for s in traj["steps"]:
        if s["type"] == "observation":
            c = calls.get(s["content"]["id"], {})
            pairs.append((c.get("tool"), c.get("input", {}), s["content"]["result"]))
    return pairs


def _numbers(text: str):
    nums = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(%)?", text):
        v = float(m.group(1))
        nums.append(v / 100 if m.group(2) else v)
    return nums


def grade_v2(task: dict, row: dict) -> dict:
    traj = row["trajectory"]
    pairs = _calls_and_results(traj)
    answer = strip_think(final_text(traj))
    tools_used = [p[0] for p in pairs]
    out = {"tool_used": task["mock_tool"] in tools_used, "n_tool_calls": len(pairs),
           "first_tool_correct": bool(pairs) and pairs[0][0] == task["mock_tool"]}

    ct = task["check_type"]
    if ct == "numeric":
        ok_vals = [r["result"] for t, _, r in pairs if t == "calculator" and r.get("ok")]
        out["success_v2"] = any(abs(v - task["expected_value"]) <= task["tolerance"] for v in ok_vals)
        # 诊断:最终回答里是否出现了正确的数(允许 2 位小数的四舍五入或百分数写法)
        out["answer_has_value"] = any(abs(v - task["expected_value"]) <= 0.006 for v in _numbers(answer))
    elif ct == "term":
        exp = task["expected_term"].lower()
        hits = [r for t, _, r in pairs if t == "concept_lookup" and r.get("ok")
                and (exp in r.get("term", "").lower() or r.get("term", "").lower() in exp)]
        out["success_v2"] = bool(hits)
        # 诊断:最终回答里是否提到了这个术语(检查"查到了但没用上"的情况)
        out["answer_mentions_term"] = exp.split("-")[0] in answer.lower()
        out["wrong_lookup"] = any(t == "concept_lookup" and r.get("ok") and r not in hits for t, _, r in pairs)
    elif ct == "tool_error_handled":
        out["success_v2"] = any(k in answer.lower() for k in FAILURE_SIGNALS)
        out["tool_attempted"] = bool(pairs)
    else:
        out["success_v2"] = False
    return out
