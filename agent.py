"""
最小 Agent demo —— 核心循环。

这个文件只做一件事:实现"推理 -> 工具调用 -> 观察 -> 再推理"的循环(ReAct 结构),
并把每一步都记录成轨迹(trajectory),对应我们之前讲的 Agent 评测里
"轨迹评估 (trajectory evaluation)" 需要的原始数据。

跑的时候分两种模式:
  - USE_MOCK_LLM = True  : 用一个假的、规则写死的"LLM"来跑通循环逻辑,不花钱、不用API key
  - USE_MOCK_LLM = False : 换成真实 Anthropic API(需要你自己的 key),
                           真正让 Claude 自主决定调不调工具、调哪个工具

你先看 Mock 模式怎么跑通,再切换到真实模式,代码结构完全不用改,
只是"大脑"从假的换成真的——这正好对应"Agent = LLM 大脑 + 工具 + 循环"这句话。
"""

import json
import os
from datetime import datetime

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MAX_STEPS = 5  # 防止死循环失控(真实场景里一定要有这个上限)


# =========================================================
# 轨迹记录器:对应"Agent 评测要先有轨迹数据才能评"
# =========================================================
class Trajectory:
    def __init__(self, task: str):
        self.task = task
        self.steps = []
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def log(self, step_type: str, content: dict):
        """
        step_type 取值: "reasoning" | "tool_call" | "observation" | "final_answer"
        这四类恰好对应 ReAct 循环的四个阶段。
        """
        self.steps.append({
            "step_index": len(self.steps),
            "type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    def to_dict(self):
        return {"task": self.task, "started_at": self.started_at, "steps": self.steps}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def print_summary(self):
        print(f"\n===== 轨迹回放: {self.task} =====")
        for s in self.steps:
            print(f"[{s['step_index']}] {s['type']}: {json.dumps(s['content'], ensure_ascii=False)}")
        print("=" * 50)


# =========================================================
# 假 LLM:规则写死,只用来验证循环的控制流是否正确
# =========================================================
def mock_llm_decide(task: str, history: list) -> dict:
    """
    模拟 LLM 的"推理"这一步。
    根据已经执行过多少步工具调用,决定下一步是调用哪个工具、还是给出最终答案。
    这不是真的智能,只是用来测试: 推理->工具调用->观察->再推理 这个循环能不能正确跑完。
    """
    tool_call_count = sum(1 for h in history if h["type"] == "tool_call")

    if "0.95" in task and "8" in task and tool_call_count == 0:
        return {
            "action": "tool_call",
            "tool": "calculator",
            "input": {"expression": "0.95 ** 8"},
            "reasoning": "任务涉及连乘计算,我需要先用计算器算出精确值,而不是自己心算。",
        }

    if "CUPED" in task.upper() and tool_call_count == 0:
        return {
            "action": "tool_call",
            "tool": "concept_lookup",
            "input": {"term": "CUPED"},
            "reasoning": "我需要先查一下 CUPED 的准确定义,确保回答准确,而不是凭印象回答。",
        }

    if tool_call_count == 0:
        # 默认示例:先查概念,再做一次计算,测试"多步循环"是否能正确推进
        return {
            "action": "tool_call",
            "tool": "concept_lookup",
            "input": {"term": "bradley-terry"},
            "reasoning": "先查一下相关术语的定义,作为回答的依据。",
        }

    if tool_call_count == 1:
        return {
            "action": "tool_call",
            "tool": "calculator",
            "input": {"expression": "0.95 ** 8"},
            "reasoning": "已经拿到概念定义了,现在再做一步数值计算,测试多工具协作。",
        }

    # 已经调用过工具、拿到观察结果了,给出最终答案
    return {
        "action": "final_answer",
        "content": "(模拟回答) 已经综合了工具返回的信息,这里给出最终结论。",
    }


def real_llm_decide(task: str, history: list, client, model: str) -> dict:
    """
    真实模式:把 task + 历史轨迹 交给 Claude,让它自己决定调用哪个工具,
    或者直接给出最终答案。用的是 Anthropic 官方的 tool use (function calling) 机制。
    """
    messages = [{"role": "user", "content": task}]

    # 把历史的 工具调用/观察 还原成多轮对话,让模型"记得"前面发生了什么
    for h in history:
        if h["type"] == "tool_call":
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": h["content"]["id"],
                    "name": h["content"]["tool"],
                    "input": h["content"]["input"],
                }],
            })
        elif h["type"] == "observation":
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": h["content"]["id"],
                    "content": json.dumps(h["content"]["result"], ensure_ascii=False),
                }],
            })

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=TOOL_SCHEMAS,
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use":
            return {
                "action": "tool_call",
                "tool": block.name,
                "input": block.input,
                "id": block.id,
                "reasoning": "(真实模式下,模型内部推理不直接可见,这里只记录它选择的工具)",
            }

    # 没有工具调用,说明模型认为可以直接回答了
    text_parts = [b.text for b in response.content if b.type == "text"]
    return {"action": "final_answer", "content": "\n".join(text_parts)}


# =========================================================
# 主循环:推理 -> 工具调用 -> 观察 -> 再推理
# =========================================================
def run_agent(task: str, use_mock: bool = True, client=None, model: str = "claude-sonnet-5",
              decide_fn=None, verbose: bool = True):
    """
    decide_fn: 可选的自定义"推理"函数,签名 (task, history) -> decision。
    评测脚本(eval/)用它来给每个任务指定一个"正确工具"的最小化假大脑,
    而不去碰 mock_llm_decide / real_llm_decide 这两个默认实现——
    这样评测逻辑和这里的循环逻辑解耦,互不影响。
    """
    trajectory = Trajectory(task)
    history = []

    if verbose:
        print(f"\n>>> 任务: {task}")

    for step in range(MAX_STEPS):
        # ---- 1. 推理 (Reasoning) ----
        if decide_fn is not None:
            decision = decide_fn(task, history)
        elif use_mock:
            decision = mock_llm_decide(task, history)
        else:
            decision = real_llm_decide(task, history, client, model)

        if decision["action"] == "final_answer":
            trajectory.log("final_answer", {"content": decision["content"]})
            if verbose:
                print(f"最终答案: {decision['content']}")
            return trajectory

        # ---- 2. 工具调用 (Tool Call) ----
        tool_name = decision["tool"]
        tool_input = decision["input"]
        call_id = decision.get("id", f"call_{step}")

        trajectory.log("reasoning", {"reasoning": decision.get("reasoning", "")})
        trajectory.log("tool_call", {"id": call_id, "tool": tool_name, "input": tool_input})
        if verbose:
            print(f"  [步骤{step}] 调用工具: {tool_name}({tool_input})")

        history.append({"type": "tool_call", "content": {"id": call_id, "tool": tool_name, "input": tool_input}})

        # ---- 3. 观察 (Observation): 真正执行工具,拿到真实结果 ----
        if tool_name not in TOOL_FUNCTIONS:
            result = {"ok": False, "error": f"未知工具: {tool_name}"}
        else:
            result = TOOL_FUNCTIONS[tool_name](**tool_input)

        trajectory.log("observation", {"id": call_id, "result": result})
        if verbose:
            print(f"  [步骤{step}] 观察结果: {result}")

        history.append({"type": "observation", "content": {"id": call_id, "result": result}})

        # ---- 4. 循环回到第1步,带着新的 history 再次推理 ----

    trajectory.log("final_answer", {"content": "(达到最大步数,强制终止——真实系统里这算一种失败模式)"})
    if verbose:
        print("已达到最大步数,强制终止循环。")
    return trajectory


# =========================================================
# 入口:先跑 Mock 模式,验证控制流程
# =========================================================
if __name__ == "__main__":
    USE_MOCK_LLM = True  # 换成 False 之前,先设好 ANTHROPIC_API_KEY 环境变量

    test_tasks = [
        "任务成功率是 0.95,连续执行 8 步,整体成功率是多少?",
        "帮我解释一下 CUPED 是什么,为什么能缩短实验周期?",
    ]

    client = None
    model = "claude-sonnet-5"
    if not USE_MOCK_LLM:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    os.makedirs("trajectories", exist_ok=True)
    for i, task in enumerate(test_tasks):
        traj = run_agent(task, use_mock=USE_MOCK_LLM, client=client, model=model)
        traj.print_summary()
        traj.save(f"trajectories/trajectory_{i}.json")
        print(f"轨迹已保存到 trajectories/trajectory_{i}.json")
