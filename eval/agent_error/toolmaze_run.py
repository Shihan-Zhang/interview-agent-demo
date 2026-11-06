"""
用 ToolMaze 官方的 run_eval.py 跑评测,但在运行时打三个补丁(不修改 ToolMaze 源码):

1. 单次请求超时:官方写死 timeout=45.0。开思考、max_tokens 16000 时单轮常超过 45 s,
   超时重试 5 次仍失败就把 "Error during execution" 当最终回答 → 接口故障被记成模型失败。
2. extra_body:把配置里 agent.extra_body(如 {"enable_thinking": true})加进每次请求,让思考模式显式可控。
3. 调用日志:每次请求写一行 JSON(token、思考 token、缓存命中、延迟、finish_reason、错误),
   用于算真实花费、区分"接口失败"和"模型失败"。

用法(在仓库根目录):
    python eval/agent_error/toolmaze_run.py --config eval/agent_error/configs/toolmaze_api_pilot.yaml --task-category c1 --limit 5
其余参数原样传给 run_eval.py。
"""
import json
import os
import runpy
import sys
import threading
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TM = os.environ.get("TOOLMAZE_DIR", os.path.join(ROOT, "external", "agent_error", "toolmaze"))


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


cfg_path = os.path.abspath(_arg("--config"))
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
patch = cfg.get("patch", {}) or {}
TIMEOUT = float(patch.get("request_timeout_s", 45))
EXTRA = cfg.get("agent", {}).get("extra_body") or None
LOG = patch.get("call_log")
if LOG:
    LOG = os.path.abspath(os.path.join(os.path.dirname(cfg_path), LOG)) if not os.path.isabs(LOG) else LOG
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
_lock = threading.Lock()

sys.path.insert(0, TM)
os.chdir(TM)
from evaluation.agents import openai_agent as oa  # noqa: E402

_orig_init = oa.OpenAIAgent.__init__


def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    create = self.client.chat.completions.create

    def wrapped(*args, **kw):
        kw["timeout"] = TIMEOUT
        if EXTRA:
            kw["extra_body"] = {**(kw.get("extra_body") or {}), **EXTRA}
        t0 = time.time()
        rec = {"ts": round(t0, 3), "model": kw.get("model"), "n_messages": len(kw.get("messages") or [])}
        try:
            r = create(*args, **kw)
            u = getattr(r, "usage", None)
            ctd = getattr(u, "completion_tokens_details", None)
            ptd = getattr(u, "prompt_tokens_details", None)
            ch = r.choices[0] if getattr(r, "choices", None) else None
            msg = getattr(ch, "message", None)
            rec.update(ok=True, latency_s=round(time.time() - t0, 2),
                       prompt_tokens=getattr(u, "prompt_tokens", None),
                       completion_tokens=getattr(u, "completion_tokens", None),
                       reasoning_tokens=getattr(ctd, "reasoning_tokens", None),
                       cached_tokens=getattr(ptd, "cached_tokens", None),
                       has_reasoning_content=bool(getattr(msg, "reasoning_content", None)),
                       finish_reason=getattr(ch, "finish_reason", None))
            return r
        except Exception as e:
            rec.update(ok=False, latency_s=round(time.time() - t0, 2), error=f"{type(e).__name__}: {str(e)[:300]}")
            raise
        finally:
            if LOG:
                with _lock, open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    self.client.chat.completions.create = wrapped


oa.OpenAIAgent.__init__ = _patched_init
print(f"[toolmaze_run] timeout={TIMEOUT}s extra_body={EXTRA} call_log={LOG}")
sys.argv = ["run_eval.py"] + [a if a != "--config" else a for a in sys.argv[1:]]
sys.argv[sys.argv.index("--config") + 1] = cfg_path
runpy.run_path(os.path.join("evaluation", "scripts", "run_eval.py"), run_name="__main__")
