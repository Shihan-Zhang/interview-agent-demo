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

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import run_agent  # noqa: E402

# 真实模式用的模型。换供应商时只改这两个常量和下面 build_real_decide_fn()
# 里对应的分支,不用动其他任何代码——这正是 decide_fn 解耦设计的意义。
NVIDIA_MODEL = "z-ai/glm-5.3"
DEEPSEEK_MODEL = "deepseek-flash"
SILICONFLOW_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # 跟 run_siliconflow_demo.py 保持一致

MODEL_BY_PROVIDER = {
    "nvidia": NVIDIA_MODEL,
    "deepseek": DEEPSEEK_MODEL,
    "siliconflow": SILICONFLOW_MODEL,
}

# 人民币/百万token 价格,供 score.py 把真实token用量换算成真实花费。
# 这是手动维护的表,不是从API动态查的——供应商随时可能调价,这个数字过时了
# 不会报错,只会让算出来的成本不准,所以表里都标了"按xx时查到的价格"方便以后核对更新。
# DeepSeek按时段/缓存命中状态有4种价格,这里保守地用"空闲时段+缓存未命中"这个最贵档,
# 不是因为它最准,而是作为上界估计更安全——真实花费大概率比这个数字低,不会比它高。
PRICING_CNY_PER_M_TOKENS = {
    # 硅基流动 Qwen/Qwen3-30B-A3B-Instruct-2507,2026-10-16 官方价格页确认
    "siliconflow": {"input": 0.7, "output": 2.8},
    # DeepSeek deepseek-flash,2026-10-16 官方价格页确认(空闲时段+缓存未命中)
    "deepseek": {"input": 1.0, "output": 4.0},
    # NVIDIA NIM 免费开发者计划,没有公开的人民币计价,先不计入成本统计
    "nvidia": None,
}


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
                "calls": [{"tool": task_spec["mock_tool"], "input": task_spec["mock_input"]}],
                "reasoning": f"(评测假大脑) 按任务标注调用 {task_spec['mock_tool']}",
            }
        # 已经有一批观察结果,如实复述第一个,不编造
        # (history里"observation"条目现在装的是列表,假大脑这里只调了一个工具,取[0]即可)
        last_obs = history[-1]["content"]["results"][0]["result"]
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
        # 没有编造一个"成功"的假结果。
        # 假大脑模式下最终答案是原样转述的JSON,所以之前直接找"false"这个子串就行;
        # 真实模型会用自然语言回答,不会输出裸JSON,所以换成关键词启发式判断——
        # 这本身是个近似方法,不是严格判定,更严谨的做法是人工抽查或接LLM-as-Judge,
        # 这里只是先给一个自动化的粗筛。
        FAILURE_KEYWORDS = [
            "错误", "失败", "无法", "不存在", "未找到", "没有找到", "找不到",
            "division by zero", "undefined", "error", "not found", "无效",
            # 以下几个是真实模型跑出来后,从实际输出里观察到、原关键词列表没覆盖到的表达——
            # 关键词式判分本来就只能"见一个补一个",覆盖不全是这种方法本身的局限,
            # 不是靠加几个词就能彻底解决的,这里先针对已观察到的case打补丁。
            "抱歉", "没有关于", "未收录", "不在我的", "未能",
        ]
        tool_failed_as_expected = bool(observations) and observations[0]["content"]["result"].get("ok") is False
        final_text = final_answers[0]["content"].get("content", "") if final_answers else ""
        did_not_fabricate_success = any(kw in final_text.lower() for kw in FAILURE_KEYWORDS) or \
                                     "ok\": false" in final_text.lower()
        outcome_correct = tool_failed_as_expected and did_not_fabricate_success
        detail = f"tool_failed_as_expected={tool_failed_as_expected}, 如实报告失败(关键词启发式,非严格判定)={did_not_fabricate_success}"

    return {
        "tool_correct": tool_correct,
        "outcome_correct": outcome_correct,
        "task_success": tool_correct and outcome_correct,
        "detail": detail,
    }


def build_real_decide_fn(provider: str):
    """
    真实模式用:构造一个指向真实大模型API的 decide_fn。注意:这个 decide_fn 是
    "无状态的"(不依赖某个特定任务),每个任务都可以复用同一个,跟
    make_eval_mock_decide() 那种"每个任务单独生成一个"不一样——
    因为真实大脑不需要被提前告知"正确答案",假大脑才需要。
    """
    from openai import OpenAI

    if provider == "nvidia":
        from providers.nvidia_provider import make_nvidia_decide_fn  # noqa: E402

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "没有找到环境变量 NVIDIA_API_KEY。"
                "Windows PowerShell 里先运行: $env:NVIDIA_API_KEY=\"你的nvapi-key\""
            )
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        return make_nvidia_decide_fn(client, NVIDIA_MODEL)

    elif provider == "deepseek":
        from providers.deepseek_provider import make_deepseek_decide_fn  # noqa: E402

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "没有找到环境变量 DEEPSEEK_API_KEY。"
                "Windows PowerShell 里先运行: $env:DEEPSEEK_API_KEY=\"你的DeepSeek key\""
            )
        client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
        return make_deepseek_decide_fn(client, DEEPSEEK_MODEL)

    elif provider == "siliconflow":
        from providers.siliconflow_provider import make_siliconflow_decide_fn  # noqa: E402

        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "没有找到环境变量 SILICONFLOW_API_KEY。"
                "Windows PowerShell 里先运行: $env:SILICONFLOW_API_KEY=\"你的key\""
            )
        client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=api_key)
        return make_siliconflow_decide_fn(client, SILICONFLOW_MODEL)

    raise ValueError(f"未知 provider: {provider}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["mock", "real"], default="mock",
        help="mock: 假大脑,只验证评测脚本本身没bug(默认)。"
             "real: 真实调用大模型API,这时候'工具选得对不对'才是真的在考察模型能力。",
    )
    parser.add_argument(
        "--provider", choices=["nvidia", "deepseek", "siliconflow"], default="siliconflow",
        help="real模式下用哪家供应商,默认siliconflow(国内直连,不需要VPN)。",
    )
    args = parser.parse_args()

    tasks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.jsonl")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.jsonl")

    tasks = load_tasks(tasks_path)
    results = []

    real_decide_fn = build_real_decide_fn(args.provider) if args.mode == "real" else None

    print(f"共 {len(tasks)} 个评测任务,模式={args.mode}"
          f"{f'(provider={args.provider})' if args.mode == 'real' else ''},开始运行...\n")

    for task_spec in tasks:
        decide_fn = real_decide_fn if args.mode == "real" else make_eval_mock_decide(task_spec)
        trajectory = run_agent(task_spec["task"], decide_fn=decide_fn, verbose=False)
        judgement = check_outcome(task_spec, trajectory)

        row = {
            "id": task_spec["id"],
            "task": task_spec["task"],
            "check_type": task_spec["check_type"],
            "mode": args.mode,
            "provider": args.provider if args.mode == "real" else None,
            "model": MODEL_BY_PROVIDER.get(args.provider) if args.mode == "real" else None,
            **judgement,
            "trajectory": trajectory.to_dict(),
        }
        results.append(row)

        status = "✅ PASS" if row["task_success"] else "❌ FAIL"
        print(f"[{task_spec['id']}] {status}  tool_correct={row['tool_correct']}  "
              f"outcome_correct={row['outcome_correct']}  ({row['detail']})")

        if args.mode == "real":
            time.sleep(1)  # 给免费API留点余量,避免短时间内触发限流(40次/分钟)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n详细结果(含完整轨迹)已保存到 {out_path}")
    return results


if __name__ == "__main__":
    main()
