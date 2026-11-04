"""判分第二版和操纵检查。用构造的轨迹,覆盖读失败轨迹时发现的几种情况。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from graders import grade_v2, manipulation_check, strip_think  # noqa: E402

TERM_TASK = {"id": "t06", "mock_tool": "concept_lookup", "check_type": "term", "expected_term": "位置偏差"}
ERR_TASK = {"id": "t12", "mock_tool": "concept_lookup", "check_type": "tool_error_handled", "expected_term": None}


def traj(steps):
    return {"task": "x", "steps": steps}


def call(i, tool, inp):
    return {"type": "tool_call", "content": {"id": i, "tool": tool, "input": inp}}


def obs(i, result):
    return {"type": "observation", "content": {"id": i, "result": result}}


def final(text):
    return {"type": "final_answer", "content": {"content": text}}


def test_retry_after_failed_first_lookup_counts_as_success():
    # 第一次用英文查询失败,第二次改用中文查到了:第一版判失败,第二版应判成功
    t = traj([call("a", "concept_lookup", {"term": "position bias"}), obs("a", {"ok": False, "error": "没找到"}),
              call("b", "concept_lookup", {"term": "位置偏差"}), obs("b", {"ok": True, "term": "位置偏差"}),
              final("位置偏差是指……")])
    g = grade_v2(TERM_TASK, {"trajectory": t})
    assert g["success_v2"]
    assert g["n_tool_calls"] == 2


def test_honest_refusal_without_tool_call_counts_as_success():
    t = traj([final("术语库中并没有收录这个术语,我无法给出定义。")])
    assert grade_v2(ERR_TASK, {"trajectory": t})["success_v2"]


def test_manipulation_check_flags_thinking_leaked_into_answer():
    leaked = {"status": "ok", "think": False, "usage": {"thinking_chars": 0},
              "trajectory": traj([final("好的,我现在需要……\n</think>\n\n答案是 7")])}
    clean = {"status": "ok", "think": False, "usage": {"thinking_chars": 0},
             "trajectory": traj([final("<think>\n\n</think>\n\n答案是 7")])}
    assert not manipulation_check(leaked)
    assert manipulation_check(clean)


def test_strip_think_keeps_only_user_facing_answer():
    assert strip_think("<think>\n\n</think>\n\n答案是 7") == "答案是 7"
    assert strip_think("推理过程……</think>答案是 7") == "答案是 7"
