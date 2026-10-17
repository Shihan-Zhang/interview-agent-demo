"""
给 run_agent() 用的"DeepSeek版大脑"。

跟 nvidia_provider.py 几乎是同一份代码——这正是当初把"循环"和"大脑"解耦这个
设计决定的价值所在:换一个供应商,不用碰 agent.py,甚至不用碰 tools.py,
只需要照着同样的结构,把 base_url 和参数细节换成新供应商的就行。

DeepSeek的API是OpenAI兼容格式,tools参数的结构跟NVIDIA那边完全一样
(都是 {type: "function", function: {name, description, parameters}}),
所以这个文件和 nvidia_provider.py 的代码几乎是复制过来改了个名字——
这不是偷懒,这恰好说明"OpenAI协议"这件事的价值:只要供应商遵循同一套协议,
切换供应商的成本就被协议本身压得很低。
"""

import json


def make_deepseek_decide_fn(client, model: str):
    """
    client: 一个指向 DeepSeek 的 openai.OpenAI 实例(base_url="https://api.deepseek.com")
    model:  模型ID,比如 "deepseek-flash"(轻量、便宜,适合这种小规模测试)

    返回一个 decide_fn,签名跟 nvidia_provider.make_nvidia_decide_fn 完全一样——
    这就是为什么 agent.py、eval/run_eval.py 都不需要知道"现在用的是哪家供应商"。
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
