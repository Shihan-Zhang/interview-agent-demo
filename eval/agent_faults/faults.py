"""
故障注入:在工具函数外面包一层,按设定在第几次调用时返回故障,而不改工具本身。

五类故障(按"Agent 看到的信号"划分,因为恢复策略取决于信号,而不是取决于故障在哪一层发生):

| 类别 | 信号 | 持续 | 期望行为 |
|---|---|---|---|
| timeout      | ok=False, retryable=True,"请求超时"             | 只有第 1 次 | 重试 → 接地成功 |
| rate_limited | ok=False, retryable=True, retry_after_s,"429" | 只有第 1 次 | 重试 → 接地成功 |
| malformed    | ok=True,但只给一段被截断的 JSON 字符串          | 只有第 1 次 | 发现残缺 → 重试 → 接地成功 |
| empty        | ok=True,结果为空(形状正常)                   | 只有第 1 次 | 不把"空"当"没有" → 换查询重试 → 接地成功 |
| unavailable  | ok=False, retryable=False,"503"               | 整道题     | 不编造;说明查不到及原因(诚实失败) |

前四类是"暂时性"的:第二次调用就正常,所以能恢复;第五类是"持续性"的,正确结果只有诚实失败。
malformed 和 empty 是"静默"故障(ok=True):最危险,因为不报错,模型可能直接据此下结论。

注入只针对每道题的目标工具(task["tool"]),其他工具不受影响。
每次注入都记进 events,评测据此区分"真实返回"和"注入的返回"。
"""

import copy
import json

FAULT_KINDS = ["timeout", "rate_limited", "malformed", "empty", "unavailable"]
TRANSIENT = {"timeout", "rate_limited", "malformed", "empty"}
CONDITIONS = ["none"] + FAULT_KINDS


def _empty_like(result: dict) -> dict:
    """把真实结果改成"形状正常但没有数据"的版本,保持字段一致,模型从形状上看不出被注入。"""
    out = copy.deepcopy(result)
    if "results" in out:
        out["results"] = []
    if "rows" in out:
        out["rows"], out["count_total"], out["returned"] = [], 0, 0
    return out


def _fault_result(kind: str, real_result: dict = None) -> dict:
    if kind == "timeout":
        return {"ok": False, "error_type": "timeout", "error": "请求超时(10 s 无响应)", "retryable": True}
    if kind == "rate_limited":
        return {"ok": False, "error_type": "rate_limited", "error": "429 Too Many Requests:请求过于频繁,请稍后重试",
                "retryable": True, "retry_after_s": 2}
    if kind == "unavailable":
        return {"ok": False, "error_type": "unavailable", "error": "503 Service Unavailable:服务暂时不可用",
                "retryable": False}
    if kind == "malformed":
        raw = json.dumps(real_result, ensure_ascii=False)
        return {"ok": True, "raw": raw[: max(20, len(raw) * 2 // 5)]}   # 截掉后 60%,JSON 不完整
    if kind == "empty":
        return _empty_like(real_result)
    raise ValueError(kind)


class FaultInjector:
    def __init__(self, tool_functions: dict, target_tool: str, kind: str, fail_calls: int = 1):
        assert kind in CONDITIONS, kind
        self.inner, self.target, self.kind = tool_functions, target_tool, kind
        self.fail_calls = fail_calls          # 暂时性故障前几次调用出错
        self.calls = 0                        # 目标工具被调用的次数
        self.events = []                      # [{call_index, tool, injected, ok}]

    def _should_inject(self) -> bool:
        if self.kind == "none":
            return False
        if self.kind == "unavailable":
            return True
        return self.calls <= self.fail_calls

    def wrap(self) -> dict:
        wrapped = {}
        for name, fn in self.inner.items():
            wrapped[name] = self._wrap_one(name, fn) if name == self.target else self._record(name, fn)
        return wrapped

    def _record(self, name, fn):
        def call(**kwargs):
            result = fn(**kwargs)
            self.events.append({"tool": name, "call_index": None, "injected": False,
                                "ok": bool(isinstance(result, dict) and result.get("ok"))})
            return result
        return call

    def _wrap_one(self, name, fn):
        def call(**kwargs):
            self.calls += 1
            if self._should_inject():
                # malformed / empty 需要先拿到真实结果再改形状;真实调用失败(如参数错)就原样返回
                if self.kind in ("malformed", "empty"):
                    real = fn(**kwargs)
                    if not (isinstance(real, dict) and real.get("ok")):
                        self.events.append({"tool": name, "call_index": self.calls, "injected": False,
                                            "ok": False})
                        return real
                    result = _fault_result(self.kind, real)
                else:
                    result = _fault_result(self.kind)
                self.events.append({"tool": name, "call_index": self.calls, "injected": self.kind, "ok": False})
                return result
            result = fn(**kwargs)
            self.events.append({"tool": name, "call_index": self.calls, "injected": False,
                                "ok": bool(isinstance(result, dict) and result.get("ok"))})
            return result
        return call

    def genuine_success(self) -> bool:
        """目标工具是否至少有一次未被注入的成功返回——"接地"的前提。"""
        return any(e["tool"] == self.target and not e["injected"] and e["ok"] for e in self.events)

    def injected_count(self) -> int:
        return sum(1 for e in self.events if e["injected"])
