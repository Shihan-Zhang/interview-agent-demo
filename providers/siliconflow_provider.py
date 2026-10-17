"""
给 run_agent() 用的"硅基流动(SiliconFlow)版大脑"。

跟 nvidia_provider.py / deepseek_provider.py 结构完全一样——OpenAI兼容协议
带来的好处:只要换个 base_url 和 model,其他代码原样复用。

官方API文档(api-docs.siliconflow.cn)确认过:
- base_url: https://api.siliconflow.cn/v1 (.cn域名,国内公司,不需要VPN)
- /chat/completions 接口支持 tools 参数,格式跟 OpenAI 一致
- 推荐模型 Qwen/Qwen3-8B,官方模型页确认支持工具调用(Tools: Supported)
"""

import json


def make_siliconflow_decide_fn(client, model: str):
    """
    client: 一个指向硅基流动的 openai.OpenAI 实例
            (base_url="https://api.siliconflow.cn/v1")
    model:  模型ID,比如 "Qwen/Qwen3-8B"

    返回一个 decide_fn,签名跟其他 provider 的 make_*_decide_fn 完全一样。
    """
    from tools import OPENAI_TOOL_SCHEMAS

    def decide_fn(task: str, history: list) -> dict:
        messages = [{"role": "user", "content": task}]

        # history里的"tool_call"条目装的是一批(可能不止一个)调用——
        # OpenAI协议要求同一轮里发起的多个tool_calls,放进同一条assistant消息的
        # tool_calls数组里,对应的多个结果,各自用一条独立的"tool"角色消息返回。
        for h in history:
            if h["type"] == "tool_call":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["tool"],
                                "arguments": json.dumps(c["input"], ensure_ascii=False),
                            },
                        }
                        for c in h["content"]["calls"]
                    ],
                })
            elif h["type"] == "observation":
                for r in h["content"]["results"]:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": r["id"],
                        "content": json.dumps(r["result"], ensure_ascii=False),
                    })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOL_SCHEMAS,
        )
        message = response.choices[0].message

        # 真实成本记录:response.usage 是这次API调用实际消耗的token数(不是估算的)——
        # OpenAI兼容接口统一用 prompt_tokens/completion_tokens/total_tokens 这三个字段名。
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        if message.tool_calls:
            # message.tool_calls 本身就是个列表——模型可能一轮决定调用不止一个工具,
            # 之前只取 [0] 等于白白丢掉了后面的调用,这里改成全部收集。
            calls = []
            for call in message.tool_calls:
                try:
                    tool_input = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                calls.append({"tool": call.function.name, "input": tool_input, "id": call.id})
            return {
                "action": "tool_call",
                "calls": calls,
                "reasoning": "(真实模型内部推理过程不直接可见,这里只记录它选择的工具)",
                "usage": usage,
            }

        return {"action": "final_answer", "content": message.content or "", "usage": usage}

    return decide_fn
