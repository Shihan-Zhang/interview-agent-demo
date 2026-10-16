"""
测试的不是"任务完成得好不好"(那是 eval/ 的事),
而是"推理->工具调用->观察->再推理"这个循环本身的控制流逻辑对不对:
- 该停的时候会不会停
- 失控的时候(假大脑一直不给最终答案)会不会被 MAX_STEPS 兜底,而不是无限循环下去
这两条对真实系统来说是"安全底线",不是锦上添花——
一个agent如果不会正确停止,接入真实业务就是故障。
"""

from agent import run_agent, MAX_STEPS


def test_agent_stops_when_decide_fn_returns_final_answer():
    def decide_fn(task, history):
        if history:
            return {"action": "final_answer", "content": "done"}
        return {
            "action": "tool_call",
            "tool": "calculator",
            "input": {"expression": "1 + 1"},
            "reasoning": "test",
        }

    trajectory = run_agent("测试任务", decide_fn=decide_fn, verbose=False)

    assert trajectory.steps[-1]["type"] == "final_answer"
    tool_calls = [s for s in trajectory.steps if s["type"] == "tool_call"]
    assert len(tool_calls) == 1  # 只应该调用一次工具就该停了,多调用说明循环逻辑有问题


def test_agent_does_not_loop_forever_when_decide_fn_never_stops():
    # 故意写一个"坏的假大脑":永远不给 final_answer,只会一直调用工具
    def runaway_decide_fn(task, history):
        return {
            "action": "tool_call",
            "tool": "calculator",
            "input": {"expression": "1 + 1"},
            "reasoning": "永远不停",
        }

    trajectory = run_agent("测试任务", decide_fn=runaway_decide_fn, verbose=False)

    tool_calls = [s for s in trajectory.steps if s["type"] == "tool_call"]
    # MAX_STEPS 应该兜底,不管假大脑多"固执",调用次数不能超过这个上限
    assert len(tool_calls) <= MAX_STEPS


def test_agent_executes_real_tool_not_fake_result():
    # 验证工具调用后拿到的是工具真实算出来的结果,不是随便编的占位文本
    def decide_fn(task, history):
        if history:
            return {"action": "final_answer", "content": "done"}
        return {
            "action": "tool_call",
            "tool": "calculator",
            "input": {"expression": "6 * 7"},
            "reasoning": "test",
        }

    trajectory = run_agent("测试任务", decide_fn=decide_fn, verbose=False)
    observations = [s for s in trajectory.steps if s["type"] == "observation"]

    assert observations[0]["content"]["result"]["result"] == 42
