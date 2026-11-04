"""本地模型 provider 的两个纯函数:消息拼装和回复解析。不需要启动 Ollama。"""

from providers.ollama_provider import build_messages, parse_response


def test_build_messages_restores_tool_round():
    history = [
        {"type": "tool_call", "content": {"calls": [
            {"id": "c1", "tool": "calculator", "input": {"expression": "0.9 ** 5"}}]}},
        {"type": "observation", "content": {"results": [
            {"id": "c1", "result": {"ok": True, "result": 0.59049}}]}},
    ]
    msgs = build_messages("算一下", history)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "calculator", "arguments": {"expression": "0.9 ** 5"}}
    assert msgs[2]["tool_name"] == "calculator"


def test_parse_response_tool_call_with_string_arguments_and_timings():
    data = {
        "message": {"role": "assistant", "content": "", "thinking": "先算一下",
                    "tool_calls": [{"function": {"name": "calculator", "arguments": "{\"expression\": \"1/0\"}"}}]},
        "prompt_eval_count": 200, "eval_count": 50,
        "prompt_eval_duration": 0.5e9, "eval_duration": 1e9, "total_duration": 1.6e9,
        "done_reason": "stop",
    }
    d = parse_response(data, step_index=0)
    assert d["action"] == "tool_call"
    assert d["calls"][0] == {"tool": "calculator", "input": {"expression": "1/0"}, "id": "call_0_0"}
    assert d["usage"]["decode_tok_per_s"] == 50
    assert d["usage"]["prefill_s"] == 0.5


def test_parse_response_final_answer():
    d = parse_response({"message": {"role": "assistant", "content": "结果是 7"}}, step_index=1)
    assert d == {"action": "final_answer", "content": "结果是 7", "usage": d["usage"]}
    assert d["usage"]["decode_tok_per_s"] is None
