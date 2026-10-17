"""
给 run_agent() 用的"NVIDIA版大脑"。

这个文件不碰 agent.py 的循环逻辑一行代码——完全靠 run_agent() 已经开好的
decide_fn 接口插进去。这就是"把循环和大脑解耦"这个设计决定开始回本的地方:
以后想换成任何别的OpenAI协议供应商(本地跑的模型、别的云厂商),
只需要再写一个类似这样的小文件,agent.py 完全不用动。
"""

import json

from tools import OPENAI_TOOL_SCHEMAS, TOOL_FUNCTIONS  # noqa: F401 (TOOL_FUNCTIONS 由 agent.py 执行工具时用)


def make_nvidia_decide_fn(client, model: str):
    """
    client: 一个指向 NVIDIA NIM 的 openai.OpenAI 实例
            (base_url="https://integrate.api.nvidia.com/v1")
    model:  模型ID,形如 "meta/llama-3.1-70b-instruct"(必须是支持 function calling 的模型,
            NVIDIA的模型目录页面上会标注"Function calling"这个标签)

    返回一个 decide_fn,签名是 (task, history) -> decision,
    和 agent.py 里 mock_llm_decide / real_llm_decide 的签名完全一样——
    这正是能够"即插即用"的原因:run_agent() 只关心这个签名,不关心背后是谁家的API。
    """

    def decide_fn(task: str, history: list) -> dict:
        messages = [{"role": "user", "content": task}]

        # 把历史的 工具调用/观察 还原成 OpenAI 格式的多轮对话,
        # 让模型"记得"前面已经调用过什么、拿到了什么结果。
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
                    # 真实模型偶尔会返回格式不完全合法的JSON参数——
                    # 这种"模型输出不严格符合约定格式"本身就是真实Agent系统里
                    # 要专门处理的失败模式之一,先用空字典兜底,不让程序直接崩溃。
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
