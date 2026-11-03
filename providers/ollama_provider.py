"""
本地模型(Ollama)版"大脑"。

为什么不像其他 provider 一样走 OpenAI 兼容接口,而是用 Ollama 自己的 /api/chat:
1. 要能开关"思考模式"(think=True/False),OpenAI 兼容接口控制不了这一项;
2. 原生接口直接返回耗时拆分(加载、prefill、decode),能算出 TTFT 和生成速度,
   这正是 M6 学的推理指标。

只用 Python 标准库(urllib),不需要额外安装依赖。
"""

import json
import urllib.request

DEFAULT_BASE_URL = "http://localhost:11434"
NS = 1e9  # Ollama 返回的耗时单位是纳秒


def build_messages(task: str, history: list, user_suffix: str = "") -> list:
    """把 agent.py 的 history 还原成 Ollama 原生格式的多轮消息(纯函数,方便单元测试)。
    user_suffix:附加在用户问题后面的文字,例如 Qwen3 的软开关 " /no_think"。"""
    messages = [{"role": "user", "content": task + user_suffix}]
    id_to_tool = {}
    for h in history:
        if h["type"] == "tool_call":
            calls = h["content"]["calls"]
            for c in calls:
                id_to_tool[c["id"]] = c["tool"]
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": c["tool"], "arguments": c["input"]}} for c in calls
                ],
            })
        elif h["type"] == "observation":
            for r in h["content"]["results"]:
                messages.append({
                    "role": "tool",
                    "tool_name": id_to_tool.get(r["id"], ""),
                    "content": json.dumps(r["result"], ensure_ascii=False),
                })
    return messages


def parse_response(data: dict, step_index: int) -> dict:
    """把 Ollama 的一次回复转成 agent.py 认识的 decision(纯函数,方便单元测试)。"""
    message = data.get("message", {})

    # 用量和耗时:这些字段 agent.py 会原样记进轨迹,汇总脚本再拿来算指标
    prompt_ns = data.get("prompt_eval_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    load_ns = data.get("load_duration", 0)
    eval_count = data.get("eval_count", 0)
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": eval_count,
        "total_tokens": data.get("prompt_eval_count", 0) + eval_count,
        "load_s": load_ns / NS,
        "prefill_s": prompt_ns / NS,                  # 近似 TTFT(不含排队,本地单请求时没有排队)
        "decode_s": eval_ns / NS,
        "total_s": data.get("total_duration", 0) / NS,
        "decode_tok_per_s": eval_count / (eval_ns / NS) if eval_ns else None,
        "thinking_chars": len(message.get("thinking") or ""),
        "done_reason": data.get("done_reason"),      # "length" 表示被 num_predict 截断
    }

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        calls = []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):                 # 有的版本返回字符串,统一成 dict
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({
                "tool": fn.get("name", ""),
                "input": args if isinstance(args, dict) else {},
                "id": tc.get("id") or f"call_{step_index}_{i}",
            })
        return {"action": "tool_call", "calls": calls,
                "reasoning": (message.get("thinking") or "")[:2000], "usage": usage}

    return {"action": "final_answer", "content": message.get("content", ""), "usage": usage}


def make_ollama_decide_fn(model: str, think: bool, options: dict,
                          base_url: str = DEFAULT_BASE_URL, timeout: int = 600,
                          keep_alive: str = "30m", user_suffix: str = ""):
    """
    返回一个 decide_fn,签名和其他 provider 完全一样:(task, history) -> decision。
    options: 采样参数等,原样传给 Ollama(temperature、top_p、top_k、presence_penalty、
             seed、num_ctx、num_predict)。
    """
    from tools import OPENAI_TOOL_SCHEMAS

    def decide_fn(task: str, history: list) -> dict:
        body = {
            "model": model,
            "messages": build_messages(task, history, user_suffix),
            "tools": OPENAI_TOOL_SCHEMAS,
            "think": think,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
        }
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        step_index = sum(1 for h in history if h["type"] == "tool_call")
        return parse_response(data, step_index)

    return decide_fn


def loaded_models(base_url: str = DEFAULT_BASE_URL) -> list:
    """调用 /api/ps,查看当前加载的模型和占用的显存(size_vram)。"""
    with urllib.request.urlopen(f"{base_url}/api/ps", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8")).get("models", [])
