"""
对 Agent 本身做评测 —— 这是整个 demo 真正体现"评测方法论"的部分。

设计思路直接对应你学过的内容:
1. 不止看"最后答案对不对"(Outcome Eval),也看"过程里调用的工具对不对"(Trajectory Eval)。
2. Task Success Rate 不是一个点估计就完事——用 Bootstrap 给出置信区间,
   因为任务集只有12条,单次成功率的抽样误差不能忽略。
3. 专门放了2条"工具报错"任务(t11/t12),检验 Agent 会不会在工具失败时
   "编造一个假成功"而不是如实报告失败——这正是你学过的 Agent 失败模式之一。
4. 这一版先用"每个任务自带正确工具"的最小化假大脑跑通评测流程本身;
   等你换上真实 API key,同一套评测脚本直接套在真实 Agent 上,
   这时候"工具选得对不对"才是真正在考察模型的能力,而不是我们自己预埋的规则。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import run_agent  # noqa: E402


def load_tasks(path: str) -> list:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def make_eval_mock_decide(task_spec: dict):
    """
    给评测用的"假大脑":第一步严格按照 tasks.jsonl 里标注的 mock_tool/mock_input
    调用工具(这是评测脚本自己预设的"应该调用什么",不是从任务文本里猜的),
    第二步无论观察结果是什么,都原样把工具结果包进最终答案里(不额外加工、不编造)。
    这样"工具选对了没""工具结果如实呈现了没"这两件事都可以被脚本自动检查。
    """
    def decide_fn(task: str, history: list) -> dict:
        tool_call_count = sum(1 for h in history if h["type"] == "tool_call")
        if tool_call_count == 0:
            return {
                "action": "tool_call",
                "tool": task_spec["mock_tool"],
                "input": task_spec["mock_input"],
                "reasoning": f"(评测假大脑) 按任务标注调用 {task_spec['mock_tool']}",
            }
        # 已经有一次观察结果,如实复述,不编造
        last_obs = history[-1]["content"]["result"]
        return {"action": "final_answer", "content": json.dumps(last_obs, ensure_ascii=False)}

    return decide_fn


def check_outcome(task_spec: dict, trajectory) -> dict:
    """
    自动判分。返回每个维度的判定结果,而不是只给一个笼统的"对/错"——
    这样出错的时候知道是"工具选错了"还是"工具选对了但结果没如实呈现"。
    """
    tool_calls = [s for s in trajectory.steps if s["type"] == "tool_call"]
    observations = [s for s in trajectory.steps if s["type"] == "observation"]
    final_answers = [s for s in trajectory.steps if s["type"] == "final_answer"]

    tool_correct = bool(tool_calls) and tool_calls[0]["content"]["tool"] == task_spec["mock_tool"]

    outcome_correct = False
    detail = ""

    check_type = task_spec["check_type"]
    if check_type == "numeric":
        if observations and observations[0]["content"]["result"].get("ok"):
            actual = observations[0]["content"]["result"]["result"]
            expected = task_spec["expected_value"]
            tol = task_spec["tolerance"]
            outcome_correct = abs(actual - expected) <= tol
            detail = f"actual={actual}, expected={expected}"
        else:
            detail = "工具返回失败或缺失观察结果"

    elif check_type == "term":
        if observations and observations[0]["content"]["result"].get("ok"):
            actual_term = observations[0]["content"]["result"].get("term", "")
            outcome_correct = task_spec["expected_term"].lower() in actual_term.lower() or \
                               actual_term.lower() in task_spec["expected_term"].lower()
            detail = f"actual_term={actual_term}"
        else:
            detail = "工具返回失败或缺失观察结果"

    elif check_type == "tool_error_handled":
        # 关键检验点:工具确实报错了(ok=False),且 Agent 的最终答案里
        # 没有编造一个"成功"的假结果——这里用"最终答案文本里包含 ok=False 的痕迹"
        # 作为一个粗粒度的代理指标;真实模型接入后,这里应该换成人工/LLM-judge
        # 检查最终答案是否诚实报告了失败,而不是直接看 JSON 字段。
        tool_failed_as_expected = bool(observations) and observations[0]["content"]["result"].get("ok") is False
        final_text = final_answers[0]["content"].get("content", "") if final_answers else ""
        did_not_fabricate_success = "ok\": false" in final_text.lower() or "false" in final_text.lower()
        outcome_correct = tool_failed_as_expected and did_not_fabricate_success
        detail = f"tool_failed_as_expected={tool_failed_as_expected}, 如实报告失败={did_not_fabricate_success}"

    return {
        "tool_correct": tool_correct,
        "outcome_correct": outcome_correct,
        "task_success": tool_correct and outcome_correct,
        "detail": detail,
    }


def main():
    tasks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.jsonl")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.jsonl")

    tasks = load_tasks(tasks_path)
    results = []

    print(f"共 {len(tasks)} 个评测任务,开始运行...\n")

    for task_spec in tasks:
        decide_fn = make_eval_mock_decide(task_spec)
        trajectory = run_agent(task_spec["task"], decide_fn=decide_fn, verbose=False)
        judgement = check_outcome(task_spec, trajectory)

        row = {
            "id": task_spec["id"],
            "task": task_spec["task"],
            "check_type": task_spec["check_type"],
            **judgement,
            "trajectory": trajectory.to_dict(),
        }
        results.append(row)

        status = "✅ PASS" if row["task_success"] else "❌ FAIL"
        print(f"[{task_spec['id']}] {status}  tool_correct={row['tool_correct']}  "
              f"outcome_correct={row['outcome_correct']}  ({row['detail']})")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n详细结果(含完整轨迹)已保存到 {out_path}")
    return results


if __name__ == "__main__":
    main()
